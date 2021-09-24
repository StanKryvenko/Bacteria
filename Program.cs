using System;
using System.Linq;

namespace Bacteria
{
    class Program
    {
        public static NeuralNetwork NeuralNet;
        
        static void Main(string[] args)
        {
            var lstInput = new[]
            {
                new[] {0.0, 1.0},
                new[] {1.0, 0.0},
                new[] {0.0, 0.0},
                new[] {1.0, 1.0},
            };
            var lstOutput = new[]
            {
                new[] {1.0},
                new[] {1.0},
                new[] {0.0},
                new[] {0.0},
            };
            NeuralNet = new NeuralNetwork();
            NeuralNet.CreateNet(new [] {2, 4, 1});
            NeuralNet.Train(lstInput.ToArray(), lstOutput.ToArray(), 1000, speed: 0.7, moment: 0.2);

            Console.WriteLine(NeuralNet.Run(new[] {0.0, 1.0})[0]);
            Console.WriteLine(NeuralNet.Run(new[] {1.0, 1.0})[0]);
            Console.WriteLine(NeuralNet.Run(new[] {1.0, 0.0})[0]);
            Console.WriteLine(NeuralNet.Run(new[] {0.0, 0.0})[0]);
        }
    }
}