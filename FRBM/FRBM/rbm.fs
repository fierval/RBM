namespace Dbl.Rbm

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open Microsoft.FSharp.Linq
open System.Collections.Generic

[<AutoOpen>]
module RBM =
    let inline dm (x : Generic.Matrix<float>) : DenseMatrix = DenseMatrix.OfMatrix(x)
    let inline dv (x : Generic.Vector<float>) : DenseVector = DenseVector.OfVector(x)
    
    type DenseVector with 
        static member Exp (x : DenseVector) = x |> Vector.map (fun e -> exp e) |> dv
        static member Multipy (x : DenseVector) (y : DenseVector) = x |> Vector.mapi (fun i e -> e * y.[i]) |> dv
        static member sigm (x : DenseVector) (a : DenseVector) = 1.0 / (1.0 + (DenseVector.Exp (DenseVector.Multipy x -a))) |> dv

    type CRbm (nVisible, nHidden, data : (float [] list)) = 
        let nVisible = nVisible
        let nHidden = nHidden
        let data = data
        let sigma = 0.2
        let epsA = 0.5
        let epsW = 0.5
        let cost = 0.00001
        let moment = 0.9

        
