using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>Which captured surface a shell cell or gap belongs to.</summary>
    public enum ShellSurfaceKind : byte
    {
        Wall = 0,
        Floor = 1,
        Ceiling = 2,
        Furniture = 3
    }

    /// <summary>
    /// One connected patch of leak faces: places where observed free space
    /// meets unknown that is connected to the outside of the scan — where
    /// passthrough shows through the mesh. The scan frontier is the largest
    /// one; a gap in a finished room is a small one.
    /// </summary>
    public readonly struct MeshHole
    {
        /// <summary>World-space centroid of the patch.</summary>
        public readonly Vector3 Center;
        /// <summary>Area in square metres (faces × voxel²).</summary>
        public readonly float AreaM2;
        /// <summary>Leak faces in the patch.</summary>
        public readonly int Faces;
        /// <summary>Rough across-size: the side of a square of the same area.</summary>
        public float ApproxWidthMetres => Mathf.Sqrt(Mathf.Max(AreaM2, 0f));

        public MeshHole(Vector3 center, float areaM2, int faces)
        {
            Center = center;
            AreaM2 = areaM2;
            Faces = faces;
        }
    }

    /// <summary>
    /// Closure of the scan from the boundary of observed free space. Every
    /// voxel is free, solid or unknown; unknown connected to the outside of
    /// the volume is exterior. Free–solid faces are the surface, free–exterior
    /// faces are leaks. A sealed room has no leaks; behind a wall never counts.
    /// Independent of any camera or scene model.
    /// </summary>
    public readonly struct MeshClosure
    {
        /// <summary>SurfaceAreaM2 / (SurfaceAreaM2 + LeakAreaM2). 1 for a sealed scan.</summary>
        public readonly float Closure;
        /// <summary>Total leak area, m² (all leak faces, including patches below the hole minimum).</summary>
        public readonly float LeakAreaM2;
        /// <summary>Surface area estimate, m²: the larger of the free–solid faces × voxel² and the surface band's
        /// voxel estimate (<c>VolumeIntegrator.FaceSurfaceAreaM2</c> / <c>BandSurfaceAreaM2</c>).</summary>
        public readonly float SurfaceAreaM2;
        /// <summary>Leak patches at or above the minimum area.</summary>
        public readonly int HoleCount;
        /// <summary>All leak faces.</summary>
        public readonly int LeakFaces;
        /// <summary>Free–exterior faces on a clip plane / room AABB / volume edge; not leaks.</summary>
        public readonly int CutFaces;
        /// <summary>Largest leak patch, or default when none.</summary>
        public readonly MeshHole LargestHole;

        public MeshClosure(float closure, float leakAreaM2, float surfaceAreaM2, int holeCount,
            int leakFaces, int cutFaces, MeshHole largestHole)
        {
            Closure = closure;
            LeakAreaM2 = leakAreaM2;
            SurfaceAreaM2 = surfaceAreaM2;
            HoleCount = holeCount;
            LeakFaces = leakFaces;
            CutFaces = cutFaces;
            LargestHole = largestHole;
        }
    }

    /// <summary>
    /// A connected patch of shell cells with no scanned surface in front of
    /// them — a place passthrough would leak through the finished mesh.
    /// Reported largest first by <see cref="RoomScanSession.CopyShellGaps"/>.
    /// </summary>
    public readonly struct ShellGap
    {
        /// <summary>World-space centre of the patch on the captured surface.</summary>
        public readonly Vector3 Center;
        /// <summary>Unit march direction: into the room for walls / floor / ceiling, into the box for furniture.</summary>
        public readonly Vector3 Normal;
        /// <summary>Number of shell cells in the patch.</summary>
        public readonly int Cells;
        /// <summary>Approximate area in square metres (cells × cell²).</summary>
        public readonly float AreaM2;
        public readonly ShellSurfaceKind Kind;

        public ShellGap(Vector3 center, Vector3 normal, int cells, float areaM2, ShellSurfaceKind kind)
        {
            Center = center;
            Normal = normal;
            Cells = cells;
            AreaM2 = areaM2;
            Kind = kind;
        }
    }
}
