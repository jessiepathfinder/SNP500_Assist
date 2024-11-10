using ILGPU;
using ILGPU.Runtime;

using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace SNP500_Assist
{
	internal static class Program
	{
		private const int kernelSize = 64;
		private const double init = 1.0 / kernelSize;
		private const double lr = 1.0 / 1024;


		private const double momentumDecayNegative = 0.999;
		private const double damping = 0.5;

		static void Main(string[] args)
		{
			if(args.Length != 3){
				Console.WriteLine("USAGE: ");
				Console.WriteLine("SNP500_Assist train [dataset] [output_model]");
				Console.WriteLine("SNP500_Assist predict [dataset] [model]");
				Console.WriteLine("Download dataset here: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks (sp500_index.csv)");
				return;
			}
			Console.WriteLine("Loading dataset...");
			string[] lines = File.ReadAllLines(args[1]);
			if(lines[0] != "Date,S&P500"){
				Console.WriteLine("INVALID dataset header!");
				return;
			}
			
			int len = lines.Length - 1;
			int truelen;
			Span<(int time, double price)> span = stackalloc (int time, double price)[len];
			for (int i = 0; i < len;)
			{
				int ip1 = i + 1;
				span[i] = Parse(lines[ip1]);
				i = ip1;
			}
			int mintime = span[0].time;
			int maxtime = span[len - 1].time;
			truelen = (maxtime - mintime) + 1;
			Span<double> span1 = stackalloc double[truelen];
			Span<double> span2 = stackalloc double[truelen];

			double prevprice = span[0].price;
			double prevlogprice = Math.Log(prevprice);
			int prevtime = 0;
			for (int i = 0; i < len; ++i)
			{
				(int mytime, double myprice) = span[i];
				mytime -= mintime;
				for(int z = prevtime; z < mytime; ++z){
					span1[z] = prevprice;
					span2[z] = prevlogprice;
				}
				span1[mytime] = myprice;
				prevtime = mytime;
				prevprice = myprice;
				prevlogprice = Math.Log(myprice);
				span2[mytime] = prevlogprice;
			}
			string mode = args[0];


			if (mode == "train")
			{
				

				Console.WriteLine("Initializing Momentum-Genetic Optimizer...");

				Span<double> bestPolicy = stackalloc double[kernelSize];
				bestPolicy.Fill(init);

				Span<double> momentum = stackalloc double[kernelSize];
				momentum.Fill(0.0);

				Span<double> delta = stackalloc double[kernelSize];
				Span<double> testPolicy = stackalloc double[kernelSize];

				Console.WriteLine("Initializing ILGPU...");
				Context context = Context.Create().AllAccelerators().ToContext();
				Device device = context.GetPreferredDevice(false);
				Accelerator accelerator = device.CreateAccelerator(context);
				AcceleratorStream acs = accelerator.DefaultStream;

				Console.WriteLine("Moving data to accelerator...");
				long tl2 = truelen * 2;
				using MemoryBuffer membuf = accelerator.AllocateRaw(tl2 + kernelSize, 8);
				ArrayView<double> av = membuf.AsArrayView<double>(0, truelen);
				av.CopyFromCPU(ref span1[0], truelen);
				ArrayView<double> av2 = membuf.AsArrayView<double>(truelen, truelen);
				ArrayView<double> av3 = membuf.AsArrayView<double>(tl2, kernelSize);




				Console.WriteLine("Compiling Discrete Causal Convolution Kernel...");
				Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>  act = accelerator.LoadAutoGroupedKernel((Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>)DiscreteCausalConvKernel);

				Console.WriteLine("Computing initial policy reward...");
				double bestReward = ComputeReward(acs, act, av, av2, av3, span2, bestPolicy);
				Console.WriteLine("Reward: " + bestReward);

				Console.WriteLine("Start training...");
				for(int p = 0; p < 100000; ++p){
					Console.WriteLine("Training iteration #" + p);
					Console.WriteLine("Making random changes...");
					MakeNormalSecureRandomDoubles(delta, lr);
					for (int i = 0; i < kernelSize; ++i)
					{
						double de = delta[i] + (momentum[i] * damping);
						delta[i] = de;
						testPolicy[i] = bestPolicy[i] + de;
					}

					Console.WriteLine("Testing random changes...");
					double reward = ComputeReward(acs, act, av, av2, av3, span2, testPolicy);
					Console.WriteLine("Reward: " + reward);
					if(reward <= bestReward){
						
						for (int i = 0; i < kernelSize; ++i)
						{
							momentum[i] *= momentumDecayNegative;

						}
					} else{
						Console.WriteLine("Committing positive change...");
						bestReward = reward;
						testPolicy.CopyTo(bestPolicy);
						for (int i = 0; i < kernelSize; ++i)
						{
							momentum[i] = delta[i];

						}
					}
				}
				Console.WriteLine("Best reward: " + bestReward);
				Console.WriteLine("Saving policy...");
				using (FileStream fileStream = new FileStream(args[2], FileMode.Create, FileAccess.Write, FileShare.None, 4096, FileOptions.WriteThrough)){
					fileStream.Write(MemoryMarshal.AsBytes(bestPolicy));
					fileStream.Flush();
				}



			}
			else if (mode == "predict")
			{
				int ksshift = truelen - kernelSize;
				int ksshift1 = ksshift - 1;

				if (ksshift1 < 0) throw new Exception("Too little data!");
				Span<double> kernel = stackalloc double[kernelSize];
				using(FileStream fileStream = new FileStream(args[2], FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.SequentialScan)){
					fileStream.ReadExactly(MemoryMarshal.AsBytes(kernel));
				}
				bool compare = Dot(kernel, span1.Slice(ksshift, kernelSize)) > Dot(kernel, span1.Slice(ksshift1, kernelSize));
				Console.WriteLine(compare ? "Verdict: uptrend" : "Verdict: downtrend");
			}
			else {
				Console.WriteLine("Unsupported function!");
			}
		}
		public static double ComputeReward(AcceleratorStream acs, Action<AcceleratorStream, Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> act, ArrayView<double> av1, ArrayView<double> av2, ArrayView<double> av3, ReadOnlySpan<double> logeval, ReadOnlySpan<double> policy){
			av3.CopyFromCPU(policy);	

			int len = logeval.Length;
			act(acs, len, av1, av3, av2);
			Span<double> span = stackalloc double[len];
			av2.CopyToCPU(span);
			double reward = 0.0;
			len -= 2;
			for(int i = 0; i < len; ){
				int t = i;

				double a = span[t];
				
				i = ++t;
				double a_ = logeval[t];
				double b = span[t];
				double b_ = logeval[t + 1];
				reward += Math.Sign(b - a) * (b_ - a_);
			}
			return reward;
		}
		public static void MakeUniformSecureRandomDoubles(Span<double> span)
		{
			int len = span.Length;
			if (len == 0)
			{
				return;
			}
			RandomNumberGenerator.Fill(MemoryMarshal.AsBytes(span));
			Span<ulong> uints = MemoryMarshal.Cast<double, ulong>(span);
			for (int i = 0; i < len; ++i)
			{
				uints[i] = (uints[i] & 0x3FFFFFFFFFFFFFFF) | 0x3FF0000000000000;

				span[i] -= 1.0;
			}
		}
		public static void MakeNormalSecureRandomDoubles(Span<double> span, double std){
			int len = span.Length;
			if (len == 0) return;
			Span<double> s1 = stackalloc double[len];
			MakeUniformSecureRandomDoubles(span);
			MakeUniformSecureRandomDoubles(s1);
			for(int i = 0; i < len; ++i){
				ref double rd = ref span[i];
				double u1 = rd;
				double u2 = 1.0 - s1[i];
				rd = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
			}
		}
		private static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b){
			int len = a.Length;
			if (b.Length != a.Length) throw new Exception("Attempted to compute dot product of vectors with mismatching length!");
			double sum = 0;
			for(int i = 0; i < len; ++i){
				sum += a[i] * b[i];
			}
			return sum;
		}
		private static (int time, double price) Parse(string line){
			return ((int)(new DateTimeOffset(int.Parse(line.AsSpan(0, 4)), int.Parse(line.AsSpan(5, 2)), int.Parse(line.AsSpan(8, 2)), 0, 0, 0, TimeSpan.Zero).ToUnixTimeMilliseconds() / 86400000), double.Parse(line.AsSpan(11)));

		}
		private static void DiscreteCausalConvKernel(Index1D index, ArrayView<double> input, ArrayView<double> kernel, ArrayView<double> output){
			long index1 = index;
			long ks = kernel.Length;
			double temp = 0.0;

			for (long i = ks - IntrinsicMath.Min(ks, index1), shifted = (index1 + i + 1) - ks; i < ks; ++i, ++shifted)
			{
				temp += input[shifted] * kernel[i];
			}
			output[index1] = temp;
		}

		private static void DiscreteCausalConv(ReadOnlySpan<double> input, ReadOnlySpan<double> kernel, Span<double> output){
			int ks = kernel.Length;
			if (ks < 1) throw new Exception("Empty kernel not accepted!");
			int len = input.Length;
			if (output.Length < len) throw new Exception("Output span too small!");

			
		}
	}

}