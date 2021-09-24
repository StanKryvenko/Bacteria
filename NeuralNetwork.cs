using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Bacteria
{
    public class Synapse
    {
        [NonSerialized] public Neuron Neuron;
        public double Weight;
        public double LastDeltaWeight;
        public double Bias;

        public Synapse(Neuron connectedNeuron, double newWeight, double newBias)
        {
            Neuron = connectedNeuron;
            Weight = newWeight;
            Bias   = newBias;
        }
    }
    
    [Serializable]
    public class Neuron
    {
        private static double SigmoidDerivative(double value) => (1 - value) * value;
        private static double Sigmoid(double value)           => 1 / (1 + Math.Exp(-value));
        private Func<double, double> Activation           { get; set; }
        private Func<double, double> ActivationDerivative { get; set; }

        public readonly int _idInLayer;
        public double Input;
        public double? ForceInput;
        public double Output => ForceInput ?? Sigmoid(Input);

        public double Delta(double[] ideals) => Synapses.Any()
            ? ActivationDerivative(Output) * Synapses.Sum(x => x.Weight * x.Neuron.Delta(ideals))
            : (ideals[_idInLayer] - Output) * ActivationDerivative(Output);
        
        public readonly List<Synapse> Synapses = new();
        
        private readonly Random _random = new();

        public Neuron(int id)
        {
            Activation           = Sigmoid;
            ActivationDerivative = SigmoidDerivative;
            _idInLayer           = id;
        }
        
        public void InitWeights(Layer nextLayer)
        {
            foreach (var neuron in nextLayer.Neurons)
                Synapses.Add(new Synapse(neuron, _random.NextDouble() * 2 - 1, 0));
        }
    }

    [Serializable]
    public class Layer
    {
        public List<Neuron> Neurons { get; set; } = new();
    }

    [Serializable]
    public class NeuralNetwork
    {
        public List<Layer> Layers { get; } = new();

        private void BackwardPass(double[] ideals, double speed, double moment)
        {
            for (var layerId = Layers.Count - 1; layerId >= 0; layerId--)
                foreach (var neuron in Layers[layerId].Neurons)
                {
                    foreach (var synapse in neuron.Synapses)
                    {
                        var delta = synapse.Neuron.Delta(ideals);
                        var grad  = neuron.Output * delta;
                        var deltaWeight = speed * grad + moment * synapse.LastDeltaWeight;
                        synapse.LastDeltaWeight = deltaWeight;
                        synapse.Weight += deltaWeight;
                        synapse.Bias += delta;
                    }
                }
        }
        
        private void ForwardPass()
        {
            foreach (var neuron in Layers.SelectMany(layer => layer.Neurons))
                neuron.Input = 0;    
            
            foreach (var neuron in Layers.SelectMany(layer => layer.Neurons))
            foreach (var synapse in neuron.Synapses)
                synapse.Neuron.Input += neuron.Output * synapse.Weight + synapse.Bias;
        }
        
        public void Recreate()
        {
            for (var i = 0; i < Layers.Count - 1; i++)
                foreach (var neuron in Layers[i].Neurons)
                    for (var j = 0; j < neuron.Synapses.Count; j++)
                        neuron.Synapses[j].Neuron = Layers[i + 1].Neurons[j];
        }

        public void CreateNet(int[] map)
        {
            foreach (var layer in map)
            {
                var newLayer = new Layer();
                for (var j = 0; j < layer; j++)
                    newLayer.Neurons.Add(new Neuron(j));
                Layers.Add(newLayer);
            }

            // Set initial weights for created neurons
            for (var i = 0; i < Layers.Count - 1; i++)
                foreach (var neuron in Layers[i].Neurons)
                    neuron.InitWeights(Layers[i + 1]);
        }

        public void Train(double[][] trainSet, double[][] trainIdeals, int maxEpochs, double speed = 0.7, double moment = 0.3)
        {
            // Go through the epochs count to teach the neural net
            int epoch;
            for (epoch = 0; epoch < maxEpochs; epoch++)
            {
                // On each epoch go through the TrainSet and update error for the next epoch
                var error = 0.0;
                for (var set = 0; set < trainSet.Length; set++)
                {
                    var ideal = trainIdeals[set];
                    // Prepare inputs for input neurons
                    for (var neuronId = 0; neuronId < Layers[0].Neurons.Count; neuronId ++)
                        Layers[0].Neurons[neuronId].ForceInput = trainSet[set][neuronId];
                    
                    // Back propagation algorithm
                    ForwardPass();
                    BackwardPass(ideal, speed, moment);
                    
                    // Error debugging
                    error += Layers.Last().Neurons.Select((neuron, neuronId) => Math.Pow(ideal[neuronId] - neuron.Output, 2)).Sum();
                }
                error /= trainSet.Length;
                if (epoch % 1000 == 0) 
                    Console.WriteLine($"Passed 1k epochs. Error: {error * 100}");
                if (error < 0.005) break;
            }
            Console.WriteLine($"Trained for: {epoch} epochs");
        }
        
        public double[] Run(double[] input)
        {
            // Prepare inputs for input neurons
            for (var neuronId = 0; neuronId < Layers[0].Neurons.Count; neuronId ++)
                Layers[0].Neurons[neuronId].ForceInput = input[neuronId];
                    
            // Back propagation algorithm
            ForwardPass();
            return Layers.Last().Neurons.Select(x => x.Output).ToArray();
        }
        
        public void Test()
        {
            // Setup neural map
            CreateNet(new []{2,2,1});

            var trainSet = new[]
            {
                new double[] { 0, 0 },
                new double[] { 1, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 1 }
            };

            var trainIdeals = new[]
            {
                new double[] {0},
                new double[] {0},
                new double[] {0},
                new double[] {1}
            };
            
            Train(trainSet, trainIdeals, 10000, 0.7, 0.3);

            while (true)
            {
                var data = Console.ReadLine()?.Split(' ').Select(double.Parse);
                Console.WriteLine(Run((data ?? Array.Empty<double>()).ToArray())[0]);
            }
        }

        public string MatrixToString(bool isLong)
        {
            var outline = new StringBuilder();
            var data   = 0;
            foreach (var layer in Layers)
            {
                var synapses = layer.Neurons.SelectMany(neuron => neuron.Synapses);
                var enumerable = synapses as Synapse[] ?? synapses.ToArray();
                foreach (var synapse in enumerable)
                {
                    outline.Append($"array.set(data{(isLong ? "Long" : "Short")}, {data}, {synapse.Bias})\n");
                    data++;
                    outline.Append($"array.set(data{(isLong ? "Long" : "Short")}, {data}, {synapse.Weight})\n");
                    data++;
                }
            }

            outline.Insert(0, $"data{(isLong ? "Long" : "Short")} = array.new_float({data + 1}, 0)\n");
            return outline.ToString();
        }
    }
}