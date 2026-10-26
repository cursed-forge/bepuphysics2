using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace BepuUtilities
{
    /// <summary>
    /// Helpers for the 4-lane paths (CursedForge fork, WN-10): <see cref="Vector{T}"/> is 128 bits wide on ARM64 NEON,
    /// and on x64 when AVX is switched off (DOTNET_EnableAVX=0, the way the parity tests reach these paths on x64).
    /// Upstream only vectorises the 8-lane AVX case and falls back to scalar loops elsewhere.
    /// </summary>
    public static class Simd4
    {
        /// <summary>
        /// True when <see cref="Vector{T}"/> has 4 float lanes and a 4x4 transpose is available in hardware.
        /// Constant after JIT, so each branch on it folds away.
        /// </summary>
        public static bool IsSupported
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Vector<float>.Count == 4 && (AdvSimd.Arm64.IsSupported || Sse.IsSupported);
        }

        /// <summary>
        /// Transposes four rows into four columns: x = (a0, b0, c0, d0), y = (a1, b1, c1, d1), z = (a2, b2, c2, d2), w = (a3, b3, c3, d3).
        /// A pure data move: no arithmetic, so the result is bit-exact on every ISA.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Transpose(Vector128<float> a, Vector128<float> b, Vector128<float> c, Vector128<float> d,
            out Vector128<float> x, out Vector128<float> y, out Vector128<float> z, out Vector128<float> w)
        {
            if (AdvSimd.Arm64.IsSupported)
            {
                //zip1/zip2 on 32-bit lanes pair the rows, then zip1/zip2 on 64-bit lanes join the pairs.
                var ab0 = AdvSimd.Arm64.ZipLow(a, b);   //a0 b0 a1 b1
                var ab1 = AdvSimd.Arm64.ZipHigh(a, b);  //a2 b2 a3 b3
                var cd0 = AdvSimd.Arm64.ZipLow(c, d);   //c0 d0 c1 d1
                var cd1 = AdvSimd.Arm64.ZipHigh(c, d);  //c2 d2 c3 d3
                x = AdvSimd.Arm64.ZipLow(ab0.AsDouble(), cd0.AsDouble()).AsSingle();
                y = AdvSimd.Arm64.ZipHigh(ab0.AsDouble(), cd0.AsDouble()).AsSingle();
                z = AdvSimd.Arm64.ZipLow(ab1.AsDouble(), cd1.AsDouble()).AsSingle();
                w = AdvSimd.Arm64.ZipHigh(ab1.AsDouble(), cd1.AsDouble()).AsSingle();
            }
            else
            {
                //_MM_TRANSPOSE4_PS.
                var t0 = Sse.UnpackLow(a, b);   //a0 b0 a1 b1
                var t1 = Sse.UnpackLow(c, d);   //c0 d0 c1 d1
                var t2 = Sse.UnpackHigh(a, b);  //a2 b2 a3 b3
                var t3 = Sse.UnpackHigh(c, d);  //c2 d2 c3 d3
                x = Sse.MoveLowToHigh(t0, t1);  //a0 b0 c0 d0
                y = Sse.MoveHighToLow(t1, t0);  //a1 b1 c1 d1
                z = Sse.MoveLowToHigh(t2, t3);  //a2 b2 c2 d2
                w = Sse.MoveHighToLow(t3, t2);  //a3 b3 c3 d3
            }
        }
    }
}
