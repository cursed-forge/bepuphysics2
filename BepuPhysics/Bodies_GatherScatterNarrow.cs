using BepuPhysics.Constraints;
using BepuUtilities;
using BepuUtilities.Memory;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using static BepuUtilities.GatherScatter;

namespace BepuPhysics
{
    //CursedForge fork, WN-10: 4-lane (NEON / SSE) versions of the body gather and scatter, plus the scalar fallbacks as public statics so
    //the parity tests can run both on the same memory. The 8-lane AVX paths in Bodies_GatherScatter.cs are untouched; these run only when
    //Vector<float>.Count == 4 (ARM64 always; x64 only with AVX disabled).
    //Layout of one BodyDynamics (32 floats, 32-byte aligned): [0..3] orientation XYZW, [4..7] position XYZ + pad, [8..11] linear XYZ + pad,
    //[12..15] angular XYZ + pad, [16..23] local inertia, [24..31] world inertia; each inertia is XX YX YY ZX ZY ZZ InverseMass pad.
    //Every 4-float block of 4 bodies is one 4x4 transpose, which is the AVX 8x8 transpose split in two. Pure data movement: bit-exact.
    public partial class Bodies
    {
        //One zeroed BodyDynamics-sized block. An empty lane points here, so every load below is a plain load with no branch and no helper
        //call (a "load or zero" helper was not inlined in the large gather on ARM: the Pi profile showed it as its own frame).
        //Never written: the scatters skip empty lanes by mask or DynamicLimit.
        static readonly unsafe float* zeroBlock = AllocateZeroBlock();

        static unsafe float* AllocateZeroBlock()
        {
            var block = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(32 * sizeof(float), 64);
            System.Runtime.InteropServices.NativeMemory.Clear(block, 32 * sizeof(float));
            return block;
        }

        /// <summary>4-lane gather of body state. Empty lanes (negative index) read as zero, like the AVX path.</summary>
        public static unsafe void GatherStateNarrow<TAccessFilter>(BodyDynamics* states, Vector<int> encodedBodyIndices, bool worldInertia,
            out Vector3Wide position, out QuaternionWide orientation, out BodyVelocityWide velocity, out BodyInertiaWide inertia)
            where TAccessFilter : unmanaged, IBodyAccessFilter
        {
            Unsafe.SkipInit(out TAccessFilter filter);
            var i0 = encodedBodyIndices[0];
            var i1 = encodedBodyIndices[1];
            var i2 = encodedBodyIndices[2];
            var i3 = encodedBodyIndices[3];
            var e0 = i0 < 0;
            var e1 = i1 < 0;
            var e2 = i2 < 0;
            var e3 = i3 < 0;
            var zero = zeroBlock;
            var s0 = e0 ? zero : (float*)(states + (i0 & BodyReferenceMask));
            var s1 = e1 ? zero : (float*)(states + (i1 & BodyReferenceMask));
            var s2 = e2 ? zero : (float*)(states + (i2 & BodyReferenceMask));
            var s3 = e3 ? zero : (float*)(states + (i3 & BodyReferenceMask));

            if (filter.GatherOrientation)
            {
                Simd4.Transpose(Vector128.Load(s0), Vector128.Load(s1), Vector128.Load(s2), Vector128.Load(s3), out var x, out var y, out var z, out var w);
                orientation.X = x.AsVector();
                orientation.Y = y.AsVector();
                orientation.Z = z.AsVector();
                orientation.W = w.AsVector();
            }
            else
            {
                Unsafe.SkipInit(out orientation);
            }
            if (filter.GatherPosition)
            {
                Simd4.Transpose(Vector128.Load(s0 + 4), Vector128.Load(s1 + 4), Vector128.Load(s2 + 4), Vector128.Load(s3 + 4), out var x, out var y, out var z, out _);
                position.X = x.AsVector();
                position.Y = y.AsVector();
                position.Z = z.AsVector();
            }
            else
            {
                Unsafe.SkipInit(out position);
            }

            Unsafe.SkipInit(out velocity);
            if (filter.AccessLinearVelocity)
            {
                Simd4.Transpose(Vector128.Load(s0 + 8), Vector128.Load(s1 + 8), Vector128.Load(s2 + 8), Vector128.Load(s3 + 8), out var x, out var y, out var z, out _);
                velocity.Linear.X = x.AsVector();
                velocity.Linear.Y = y.AsVector();
                velocity.Linear.Z = z.AsVector();
            }
            if (filter.AccessAngularVelocity)
            {
                Simd4.Transpose(Vector128.Load(s0 + 12), Vector128.Load(s1 + 12), Vector128.Load(s2 + 12), Vector128.Load(s3 + 12), out var x, out var y, out var z, out _);
                velocity.Angular.X = x.AsVector();
                velocity.Angular.Y = y.AsVector();
                velocity.Angular.Z = z.AsVector();
            }

            Unsafe.SkipInit(out inertia);
            if (filter.GatherInertiaTensor || filter.GatherMass)
            {
                var offset = worldInertia ? 24 : 16;
                if (filter.GatherInertiaTensor)
                {
                    Simd4.Transpose(Vector128.Load(s0 + offset), Vector128.Load(s1 + offset), Vector128.Load(s2 + offset), Vector128.Load(s3 + offset),
                        out var xx, out var yx, out var yy, out var zx);
                    inertia.InverseInertiaTensor.XX = xx.AsVector();
                    inertia.InverseInertiaTensor.YX = yx.AsVector();
                    inertia.InverseInertiaTensor.YY = yy.AsVector();
                    inertia.InverseInertiaTensor.ZX = zx.AsVector();
                }
                Simd4.Transpose(Vector128.Load(s0 + offset + 4), Vector128.Load(s1 + offset + 4), Vector128.Load(s2 + offset + 4), Vector128.Load(s3 + offset + 4),
                    out var zy, out var zz, out var mass, out _);
                if (filter.GatherInertiaTensor)
                {
                    inertia.InverseInertiaTensor.ZY = zy.AsVector();
                    inertia.InverseInertiaTensor.ZZ = zz.AsVector();
                }
                if (filter.GatherMass)
                {
                    inertia.InverseMass = mass.AsVector();
                }
            }
        }

        /// <summary>Scalar gather, the upstream fallback. Empty lanes are left unwritten.</summary>
        public static unsafe void GatherStateFallback(BodyDynamics* states, Vector<int> encodedBodyIndices, bool worldInertia,
            out Vector3Wide position, out QuaternionWide orientation, out BodyVelocityWide velocity, out BodyInertiaWide inertia)
        {
            Unsafe.SkipInit(out position);
            Unsafe.SkipInit(out orientation);
            Unsafe.SkipInit(out velocity);
            Unsafe.SkipInit(out inertia);
            FallbackGatherMotionState(states, encodedBodyIndices, ref position, ref orientation, ref velocity);
            FallbackGatherInertia(states, encodedBodyIndices, ref inertia, worldInertia ? 24 : 16);
        }

        /// <summary>4-lane pose scatter. Like the AVX path, it writes the position pad (with Z).</summary>
        public static unsafe void ScatterPoseNarrow(BodyDynamics* states, ref Vector3Wide position, ref QuaternionWide orientation, Vector<int> encodedBodyIndices, Vector<int> mask)
        {
            Simd4.Transpose(orientation.X.AsVector128(), orientation.Y.AsVector128(), orientation.Z.AsVector128(), orientation.W.AsVector128(),
                out var o0, out var o1, out var o2, out var o3);
            Simd4.Transpose(position.X.AsVector128(), position.Y.AsVector128(), position.Z.AsVector128(), position.Z.AsVector128(),
                out var p0, out var p1, out var p2, out var p3);
            if (mask[0] != 0) { var t = (float*)(states + encodedBodyIndices[0]); Vector128.Store(o0, t); Vector128.Store(p0, t + 4); }
            if (mask[1] != 0) { var t = (float*)(states + encodedBodyIndices[1]); Vector128.Store(o1, t); Vector128.Store(p1, t + 4); }
            if (mask[2] != 0) { var t = (float*)(states + encodedBodyIndices[2]); Vector128.Store(o2, t); Vector128.Store(p2, t + 4); }
            if (mask[3] != 0) { var t = (float*)(states + encodedBodyIndices[3]); Vector128.Store(o3, t); Vector128.Store(p3, t + 4); }
        }

        /// <summary>Scalar pose scatter, the upstream fallback.</summary>
        public static unsafe void ScatterPoseFallback(BodyDynamics* states, ref Vector3Wide position, ref QuaternionWide orientation, Vector<int> encodedBodyIndices, Vector<int> mask)
        {
            for (int innerIndex = 0; innerIndex < Vector<int>.Count; ++innerIndex)
            {
                if (mask[innerIndex] == 0)
                    continue;
                ref var pose = ref states[encodedBodyIndices[innerIndex]].Motion.Pose;
                pose.Position = new Vector3(position.X[innerIndex], position.Y[innerIndex], position.Z[innerIndex]);
                pose.Orientation = new Quaternion(orientation.X[innerIndex], orientation.Y[innerIndex], orientation.Z[innerIndex], orientation.W[innerIndex]);
            }
        }

        /// <summary>4-lane scatter into the world inertia. Like the AVX path, it writes the pad (with the inverse mass).</summary>
        public static unsafe void ScatterInertiaNarrow(BodyDynamics* states, ref BodyInertiaWide inertia, Vector<int> encodedBodyIndices, Vector<int> mask)
        {
            ref var tensor = ref inertia.InverseInertiaTensor;
            Simd4.Transpose(tensor.XX.AsVector128(), tensor.YX.AsVector128(), tensor.YY.AsVector128(), tensor.ZX.AsVector128(),
                out var a0, out var a1, out var a2, out var a3);
            Simd4.Transpose(tensor.ZY.AsVector128(), tensor.ZZ.AsVector128(), inertia.InverseMass.AsVector128(), inertia.InverseMass.AsVector128(),
                out var b0, out var b1, out var b2, out var b3);
            if (mask[0] != 0) { var t = (float*)(states + encodedBodyIndices[0]) + 24; Vector128.Store(a0, t); Vector128.Store(b0, t + 4); }
            if (mask[1] != 0) { var t = (float*)(states + encodedBodyIndices[1]) + 24; Vector128.Store(a1, t); Vector128.Store(b1, t + 4); }
            if (mask[2] != 0) { var t = (float*)(states + encodedBodyIndices[2]) + 24; Vector128.Store(a2, t); Vector128.Store(b2, t + 4); }
            if (mask[3] != 0) { var t = (float*)(states + encodedBodyIndices[3]) + 24; Vector128.Store(a3, t); Vector128.Store(b3, t + 4); }
        }

        /// <summary>Scalar world inertia scatter, the upstream fallback.</summary>
        public static unsafe void ScatterInertiaFallback(BodyDynamics* states, ref BodyInertiaWide inertia, Vector<int> encodedBodyIndices, Vector<int> mask)
        {
            for (int innerIndex = 0; innerIndex < Vector<int>.Count; ++innerIndex)
            {
                if (mask[innerIndex] == 0)
                    continue;
                ref var target = ref states[encodedBodyIndices[innerIndex]].Inertia.World;
                target.InverseInertiaTensor.XX = inertia.InverseInertiaTensor.XX[innerIndex];
                target.InverseInertiaTensor.YX = inertia.InverseInertiaTensor.YX[innerIndex];
                target.InverseInertiaTensor.YY = inertia.InverseInertiaTensor.YY[innerIndex];
                target.InverseInertiaTensor.ZX = inertia.InverseInertiaTensor.ZX[innerIndex];
                target.InverseInertiaTensor.ZY = inertia.InverseInertiaTensor.ZY[innerIndex];
                target.InverseInertiaTensor.ZZ = inertia.InverseInertiaTensor.ZZ[innerIndex];
                target.InverseMass = inertia.InverseMass[innerIndex];
            }
        }

        /// <summary>
        /// 4-lane velocity scatter. Like the AVX path: when exactly one of linear and angular is accessed, only that 4-float block is written;
        /// otherwise both blocks (pads written with Z). Lanes that are not dynamic (index at or above <see cref="DynamicLimit"/>) are skipped.
        /// </summary>
        public static unsafe void ScatterVelocitiesNarrow<TAccessFilter>(BodyDynamics* states, ref BodyVelocityWide sourceVelocities, ref Vector<int> encodedBodyIndices)
            where TAccessFilter : unmanaged, IBodyAccessFilter
        {
            Unsafe.SkipInit(out TAccessFilter filter);
            var indices = (uint*)Unsafe.AsPointer(ref encodedBodyIndices);
            if (filter.AccessLinearVelocity ^ filter.AccessAngularVelocity)
            {
                int targetOffset;
                ref var source = ref sourceVelocities.Linear;
                if (filter.AccessLinearVelocity)
                {
                    targetOffset = 8;
                }
                else
                {
                    targetOffset = 12;
                    source = ref sourceVelocities.Angular;
                }
                Simd4.Transpose(source.X.AsVector128(), source.Y.AsVector128(), source.Z.AsVector128(), source.Z.AsVector128(),
                    out var r0, out var r1, out var r2, out var r3);
                if (indices[0] < DynamicLimit) Vector128.Store(r0, (float*)(states + indices[0]) + targetOffset);
                if (indices[1] < DynamicLimit) Vector128.Store(r1, (float*)(states + indices[1]) + targetOffset);
                if (indices[2] < DynamicLimit) Vector128.Store(r2, (float*)(states + indices[2]) + targetOffset);
                if (indices[3] < DynamicLimit) Vector128.Store(r3, (float*)(states + indices[3]) + targetOffset);
            }
            else
            {
                ref var linear = ref sourceVelocities.Linear;
                ref var angular = ref sourceVelocities.Angular;
                Simd4.Transpose(linear.X.AsVector128(), linear.Y.AsVector128(), linear.Z.AsVector128(), linear.Z.AsVector128(),
                    out var l0, out var l1, out var l2, out var l3);
                Simd4.Transpose(angular.X.AsVector128(), angular.Y.AsVector128(), angular.Z.AsVector128(), angular.Z.AsVector128(),
                    out var a0, out var a1, out var a2, out var a3);
                if (indices[0] < DynamicLimit) { var t = (float*)(states + indices[0]) + 8; Vector128.Store(l0, t); Vector128.Store(a0, t + 4); }
                if (indices[1] < DynamicLimit) { var t = (float*)(states + indices[1]) + 8; Vector128.Store(l1, t); Vector128.Store(a1, t + 4); }
                if (indices[2] < DynamicLimit) { var t = (float*)(states + indices[2]) + 8; Vector128.Store(l2, t); Vector128.Store(a2, t + 4); }
                if (indices[3] < DynamicLimit) { var t = (float*)(states + indices[3]) + 8; Vector128.Store(l3, t); Vector128.Store(a3, t + 4); }
            }
        }

        /// <summary>Scalar velocity scatter, the upstream fallback (writes both linear and angular).</summary>
        public static unsafe void ScatterVelocitiesFallback(BodyDynamics* states, ref BodyVelocityWide sourceVelocities, ref Vector<int> encodedBodyIndices)
        {
            var indices = (uint*)Unsafe.AsPointer(ref encodedBodyIndices);
            for (int innerIndex = 0; innerIndex < Vector<int>.Count; ++innerIndex)
            {
                if (indices[innerIndex] >= DynamicLimit)
                    continue;
                ref var sourceSlot = ref GetOffsetInstance(ref sourceVelocities, innerIndex);
                ref var target = ref states[indices[innerIndex]].Motion.Velocity;
                target.Linear = new Vector3(sourceSlot.Linear.X[0], sourceSlot.Linear.Y[0], sourceSlot.Linear.Z[0]);
                target.Angular = new Vector3(sourceSlot.Angular.X[0], sourceSlot.Angular.Y[0], sourceSlot.Angular.Z[0]);
            }
        }

        /// <summary>4-lane version of <see cref="TransposeMotionStates"/>. Lanes past the buffer length read as zero, like the AVX path.</summary>
        public static unsafe void TransposeMotionStatesNarrow(Buffer<MotionState> states, out Vector3Wide position, out QuaternionWide orientation, out BodyVelocityWide velocity)
        {
            var zero = zeroBlock;
            var s0 = (float*)states.Memory;
            var s1 = states.Length <= 1 ? zero : (float*)(states.Memory + 1);
            var s2 = states.Length <= 2 ? zero : (float*)(states.Memory + 2);
            var s3 = states.Length <= 3 ? zero : (float*)(states.Memory + 3);
            Simd4.Transpose(Vector128.Load(s0), Vector128.Load(s1), Vector128.Load(s2), Vector128.Load(s3), out var ox, out var oy, out var oz, out var ow);
            orientation.X = ox.AsVector();
            orientation.Y = oy.AsVector();
            orientation.Z = oz.AsVector();
            orientation.W = ow.AsVector();
            Simd4.Transpose(Vector128.Load(s0 + 4), Vector128.Load(s1 + 4), Vector128.Load(s2 + 4), Vector128.Load(s3 + 4), out var px, out var py, out var pz, out _);
            position.X = px.AsVector();
            position.Y = py.AsVector();
            position.Z = pz.AsVector();
            Unsafe.SkipInit(out velocity);
            Simd4.Transpose(Vector128.Load(s0 + 8), Vector128.Load(s1 + 8), Vector128.Load(s2 + 8), Vector128.Load(s3 + 8), out var lx, out var ly, out var lz, out _);
            velocity.Linear.X = lx.AsVector();
            velocity.Linear.Y = ly.AsVector();
            velocity.Linear.Z = lz.AsVector();
            Simd4.Transpose(Vector128.Load(s0 + 12), Vector128.Load(s1 + 12), Vector128.Load(s2 + 12), Vector128.Load(s3 + 12), out var ax, out var ay, out var az, out _);
            velocity.Angular.X = ax.AsVector();
            velocity.Angular.Y = ay.AsVector();
            velocity.Angular.Z = az.AsVector();
        }

        /// <summary>Scalar version of <see cref="TransposeMotionStates"/>, the upstream fallback. Lanes past the buffer length are left unwritten.</summary>
        public static void TransposeMotionStatesFallback(Buffer<MotionState> states, out Vector3Wide position, out QuaternionWide orientation, out BodyVelocityWide velocity)
        {
            Unsafe.SkipInit(out position);
            Unsafe.SkipInit(out orientation);
            Unsafe.SkipInit(out velocity);
            for (int i = 0; i < states.Length; ++i)
            {
                ref var state = ref states[i];
                Vector3Wide.WriteSlot(state.Pose.Position, i, ref position);
                QuaternionWide.WriteSlot(state.Pose.Orientation, i, ref orientation);
                Vector3Wide.WriteSlot(state.Velocity.Linear, i, ref velocity.Linear);
                Vector3Wide.WriteSlot(state.Velocity.Angular, i, ref velocity.Angular);
            }
        }
    }
}
