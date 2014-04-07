namespace Utils

open System
open System.Linq
open Combinatorics.Collections

[<AutoOpen>]
module dataGen =
    /// generate an array of arrays of random data
    /// where the majority of elements is set to 'x'
    /// len - length of each data array
    /// total - total number of such arrays
    /// return as a 2D array.
    let generateWithMajority x len total =
        if x <= 0 then failwith "repeating number should be positive"
        let rnd = Random(int DateTime.Now.Ticks)

        let major = len / 2 + 1

        // Generate a list
        let allPermutations = 
            [
                for i = major to len do
                    let xs = (Array.init len (fun j -> if j < i then x else 0)).ToList()
                    let p = Permutations<int>(xs, GenerateOption.WithoutRepetition)
                    yield! p
            ]

        let res = 
            [|
                for i = 0 to total - 1 do
                    let perm = allPermutations.[i % allPermutations.Length]
                    let changeable = perm.Select(fun e -> if e = x then e else rnd.Next(x, x + 10))
                    yield changeable.ToArray()
            |]
        Array2D.init total len (fun i j -> res.[i].[j])

