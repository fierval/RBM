namespace Dbl.Rbm

//open MathNet.Numerics.LinearAlgebra
//open MathNet.Numerics.LinearAlgebra.Double
open Fmat.Numerics
open Fmat.Numerics.MatrixFunctions
open Fmat.Numerics.BasicStat
open Fmat.Numerics.LinearAlgebra

open System.Threading
open System.Threading.Tasks
open System.Linq

open Microsoft.FSharp.Linq
open System.Collections.Generic

[<AutoOpen>]
module RBM =
    
    let memberwiseMult (x : Matrix) (y: Matrix)  =
        Matrix.applyFun2Arg (x, y, fun x1 x2 -> x1 * x2)
    
    // uniform random for sampling
    let rand () = (rand [1;1]).[1,1]
    
    type Rbm (nHidden, data : (float [] list)) = 
        // length of the visible units vector
        let nVisible = if data = Unchecked.defaultof<float [] list> || data.Length = 0 then failwith "0 visible units" else data.First().Length
        let nHidden = if nHidden > 0 then nHidden else failwith "0 hidden units"

        // data - a list of boolean vectors to represent the data
        let mutable data = data |> List.map (fun x -> Matrix(Array2D.init 1 data.Length (fun i j -> data.[i].[j])))
        let sigma = 0.2
        let epsA = 0.5
        let epsW = 0.5
        let cost = 0.00001
        let moment = 0.9

        let sigm (x : Matrix) = 1.0 / (1.0 + exp(-x))

        // weight matrix W[j,k]: j - hidden, k - visible (# of units)
        let weights = Matrix.normalRnd(0.0, 1.0, [nVisible; nHidden])
        // hidden bias, b[j]
        let hiddenBias = Matrix.normalRnd(0.0, 1.0, [1; nHidden])
        // visible bias, c[k]
        let visibleBias = Matrix.normalRnd(0.0, 1.0, [1; nVisible])
        
        let probab (bias : float) (weights : Matrix) (units : Matrix) =
            sigm(bias + weights * units).[0,0]
        
        // vector of probabilities for each hidden neuron j p(h[j] = 1 | x)
        let probabHiddenGivenVisible (visible : Matrix) = 
            Array.init nHidden (fun j -> probab hiddenBias.[0, j] weights.[j..j, 0..nHidden - 1] visible)

        // vector of probabilites for each visible neuron k p(x[k] = 1 | h)
        let probabVisibleGivenHidden (hidden : Matrix) = 
            Array.init nVisible (fun  k -> probab visibleBias.[0, k] weights.[0..nVisible - 1, k..k].T hidden)
            
        // perform Gibbs sampling for the number of steps,
        // starting with the 
        member x.bernoulliGibbsSample steps =
            let oneLayerToAnother nAnotherUnits (probFunc : Matrix -> float []) d = 
                d 
                |> List.map (fun e ->
                    let probs = probFunc e
                    let uniformSample = [|for i = 1 to e.Length do yield rand()|]
                    let newVector = Array.fold2 (fun state rnd p -> if p > rnd then state @ [1.0] else state @ [0.0]) List.empty uniformSample probs
                    let toMatrix = Matrix(Array2D.init 1 nAnotherUnits (fun i j -> newVector.[j]))
                    toMatrix
                    )

            let visibleToHidden = oneLayerToAnother nHidden probabVisibleGivenHidden
            let hiddenToVisible = oneLayerToAnother nVisible probabHiddenGivenVisible

            let rec performSteps visible hidden step = 

                if step = 0 then visible, hidden
                else
                    let hidden = visibleToHidden visible
                    let visible = hiddenToVisible hidden
                    let hidden =  visibleToHidden visible
                    performSteps visible hidden (step - 1)

            performSteps data Unchecked.defaultof<Matrix list> steps
