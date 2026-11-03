using System;
using System.Collections.Generic;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace Genesis.RoomScan
{
    /// <summary>
    /// Surface type classification for mesh vertices.
    /// </summary>
    public enum SurfaceType : byte
    {
        Unknown   = 0,
        Floor     = 1,
        Ceiling   = 2,
        Wall      = 3,
        Furniture = 4
    }

    /// <summary>
    /// Wraps Meta MRUK APIs to provide semantic room understanding.
    /// Occupancy, wall faces, and classification all live here — hosts
    /// should still prefer <see cref="RoomScanSession"/> rather than
    /// taking an MRUK dependency. Falls back to vertex-normal heuristics
    /// when MRUK room data is unavailable.
    /// </summary>
    public class RoomUnderstanding : MonoBehaviour, IRoomScanModule
    {
        public static RoomUnderstanding Instance { get; private set; }

        public string ModuleName => "Room Understanding";
        public void OnModuleInitialize(RoomScanner scanner) { }

        private MRUKRoom _room;
        private MRUK _mruk;
        private bool _subscribedToRoomEvents;

        void Awake()
        {
            Instance = this;
        }

        void OnEnable()
        {
            EnsureMruk();
            SubscribeToRoomEvents();
        }

        void OnDisable() => UnsubscribeFromRoomEvents();

        /// <summary>
        /// Raised when MRUK anchors change (created, updated, or room updated).
        /// RoomScanner subscribes to this to re-populate the SceneObjectRegistry reactively
        /// instead of polling.
        /// </summary>
        public event Action AnchorsChanged;

        // Cached per-vertex classification
        private SurfaceType[] _lastClassification;

        // ─────────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Classify a single world-space position into a <see cref="SurfaceType"/>.
        /// </summary>
        public SurfaceType GetSurfaceType(Vector3 worldPos)
        {
            EnsureRoom();
            if (_room == null) return SurfaceType.Unknown;

            MRUKAnchor best = null;
            float bestDist = float.MaxValue;

            foreach (var anchor in _room.Anchors)
            {
                float d = Vector3.Distance(anchor.transform.position, worldPos);
                if (d < bestDist) { bestDist = d; best = anchor; }
            }

            return best != null ? ClassifyAnchor(best) : SurfaceType.Unknown;
        }

        /// <summary>
        /// Bulk classification: returns a <see cref="SurfaceType"/> per vertex.
        /// When MRUK data is available, each vertex is matched to the nearest
        /// labelled anchor. Otherwise, uses vertex normal heuristics.
        /// </summary>
        public SurfaceType[] GetPerVertexSurfaceTypes(Mesh mesh)
        {
            if (mesh == null) return null;
            var verts = mesh.vertices;
            var normals = mesh.normals;
            var result = new SurfaceType[verts.Length];

            EnsureRoom();
            if (_room != null && _room.Anchors.Count > 0)
                ClassifyFromMRUK(verts, result);
            else
                ClassifyFromNormals(normals, result);

            _lastClassification = result;
            return result;
        }

        /// <summary>Returns the last computed classification, or null.</summary>
        public SurfaceType[] LastClassification => _lastClassification;

        /// <summary>All wall planes from MRUK, or empty if unavailable.</summary>
        public List<Plane> GetWallPlanes()
        {
            var planes = new List<Plane>();
            EnsureRoom();
            if (_room == null) return planes;

            foreach (var anchor in _room.Anchors)
            {
                if (!IsWall(anchor)) continue;
                var t = anchor.transform;
                planes.Add(new Plane(t.forward, t.position));
            }
            return planes;
        }

        /// <summary>
        /// Floor plane from MRUK, or a default Y=0 plane.
        ///
        /// <para>The normal is the anchor's <c>forward</c>, which is how MRUK
        /// orients a plane anchor — same convention <see cref="GetWallPlanes"/>
        /// uses. A floor anchor is therefore pitched ninety degrees so that its
        /// forward points at the ceiling, and its <c>up</c> lies flat along the
        /// floor. Reading <c>up</c> instead returns a plane standing on edge:
        /// it still passes through the right point, so anything asking whether
        /// a point is on the floor gets a plausible answer, while anything
        /// asking how high the floor is gets the height of the room's origin.
        /// </para>
        /// </summary>
        public Plane GetFloorPlane()
        {
            EnsureRoom();
            if (_room != null && _room.FloorAnchors != null && _room.FloorAnchors.Count > 0)
            {
                var ft = _room.FloorAnchors[0].transform;
                Vector3 normal = ft.forward;
                if (Vector3.Dot(normal, Vector3.up) < 0f) normal = -normal;
                return new Plane(normal, ft.position);
            }
            return new Plane(Vector3.up, Vector3.zero);
        }

        /// <summary>Bounding boxes for all furniture anchors.</summary>
        public List<Bounds> GetFurnitureBounds()
        {
            var result = new List<Bounds>();
            EnsureRoom();
            if (_room == null) return result;

            foreach (var anchor in _room.Anchors)
            {
                if (!IsFurniture(anchor)) continue;
                if (anchor.VolumeBounds.HasValue)
                    result.Add(anchor.VolumeBounds.Value);
                else if (anchor.PlaneRect.HasValue)
                {
                    var r = anchor.PlaneRect.Value;
                    var center = anchor.transform.TransformPoint(r.center);
                    result.Add(new Bounds(center, new Vector3(r.width, 0.1f, r.height)));
                }
            }
            return result;
        }

        // ─────────────────────────────────────────────────────────────
        //  Occupancy + wall faces (multi-room; never GetCurrentRoom)
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// True when the headset is inside <b>any</b> loaded captured space
        /// (outer wall planes, including doorway faces). Boot / Space Setup:
        /// any set-up room is enough. A loaded scan is tied to one room —
        /// use <see cref="IsHeadsetInsideRoom"/>. Native
        /// <c>IsPositionInRoom</c> is the floor outline and stays true a
        /// little past a doorway. <c>GetCurrentRoom()</c> is last/first after
        /// you leave — do not use it. Editor returns true.
        /// </summary>
        public bool IsHeadsetInsideAnyRoom()
        {
            if (Application.isEditor) return true;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null || _mruk.Rooms.Count == 0)
                return false;
            return Query.FindContaining(_mruk.Rooms, Query.HeadsetWorldPosition()) != null;
        }

        /// <summary>
        /// True when the headset is inside the captured space with this
        /// Scene API room UUID. False when the UUID is empty, the room is
        /// not in the loaded scene model, or the headset has left that
        /// room — even if another captured room still contains the
        /// headset. Editor returns true.
        /// </summary>
        public bool IsHeadsetInsideRoom(Guid sceneRoomUuid)
        {
            if (Application.isEditor) return true;
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            var room = Query.FindByUuid(_mruk.Rooms, sceneRoomUuid);
            return Query.Contains(room, Query.HeadsetWorldPosition());
        }

        /// <summary>True when a loaded MRUK room still has this Scene API UUID.</summary>
        public bool HasRoom(Guid sceneRoomUuid)
        {
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            return Query.FindByUuid(_mruk.Rooms, sceneRoomUuid) != null;
        }

        /// <summary>
        /// Scene API UUID of the loaded room that contains
        /// <paramref name="worldPos"/> (wall-plane test), or
        /// <see cref="Guid.Empty"/>.
        /// </summary>
        public Guid TryGetRoomUuidAt(Vector3 worldPos)
        {
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null)
                return Guid.Empty;
            return Query.RoomUuid(Query.FindContaining(_mruk.Rooms, worldPos));
        }

        /// <summary>Scene API UUID of the loaded room that contains the headset, or empty.</summary>
        public Guid TryGetRoomUuidContainingHeadset()
            => TryGetRoomUuidAt(Query.HeadsetWorldPosition());

        /// <summary>
        /// Visible <c>WALL_FACE</c> / <c>SCREEN</c> planes of every loaded
        /// room that contains <paramref name="worldPos"/>.
        /// <paramref name="kind"/> selects which labels to include.
        /// Overlapping captures are unioned. Returns 0 in the editor and
        /// when the point is not inside a captured room. Clears
        /// <paramref name="dest"/>.
        /// </summary>
        public int CopyWallFacesOfRoomContaining(
            Vector3 worldPos, List<SceneWallFace> dest, SceneFaceKind kind)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (Application.isEditor) return 0;
            EnsureMruk();
            SubscribeToRoomEvents();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyWallFacesAt(_mruk.Rooms, worldPos, dest, kind);
        }

        /// <summary>
        /// Visible vertical planes of every loaded room that contains the
        /// headset. <paramref name="kind"/> selects which labels to include.
        /// </summary>
        public int CopyHeadsetRoomWallFaces(List<SceneWallFace> dest, SceneFaceKind kind)
            => CopyWallFacesOfRoomContaining(Query.HeadsetWorldPosition(), dest, kind);

        /// <summary>
        /// Scene API labels of the anchors in every loaded room that contains
        /// <paramref name="worldPos"/> (the names of
        /// <see cref="MRUKAnchor.SceneLabels"/>: <c>BED</c>, <c>TABLE</c>,
        /// <c>COUCH</c>, <c>WALL_FACE</c>, …), each once. Overlapping captures
        /// are unioned. Returns 0 in the editor and when the point is not
        /// inside a captured room. Clears <paramref name="dest"/>.
        /// </summary>
        public int CopyLabelsOfRoomContaining(Vector3 worldPos, List<string> dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (Application.isEditor) return 0;
            EnsureMruk();
            SubscribeToRoomEvents();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyLabelsAt(_mruk.Rooms, worldPos, dest);
        }

        /// <summary>
        /// Scene API labels of every loaded room that contains the headset
        /// (see <see cref="CopyLabelsOfRoomContaining"/>).
        /// </summary>
        public int CopyHeadsetRoomLabels(List<string> dest)
            => CopyLabelsOfRoomContaining(Query.HeadsetWorldPosition(), dest);

        /// <summary>
        /// True when <paramref name="worldPos"/> is inside the captured
        /// space with this Scene API room UUID (floor outline + outer
        /// walls). False when the UUID is empty or the room is not loaded.
        /// </summary>
        public bool Contains(Guid sceneRoomUuid, Vector3 worldPos)
        {
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            return Query.Contains(Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), worldPos);
        }

        /// <summary>
        /// Half-spaces that bound <paramref name="sceneRoomUuid"/> for GPU
        /// TSDF clip: each <c>Vector4(n, w)</c> is outside when
        /// <c>dot(pos, n) &lt; w</c>. Outer walls (including doorway
        /// <c>INVISIBLE_WALL_FACE</c>), floor, and ceiling are expanded
        /// <b>outward</b> by 50 cm once so a false ceiling / plaster /
        /// depth noise still integrates. Occupancy's 8 cm inset is unchanged.
        /// Clears <paramref name="dest"/>. Returns 0 when the room is missing.
        /// </summary>
        public int CopyRoomClipPlanes(Guid sceneRoomUuid, List<Vector4> dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (sceneRoomUuid == Guid.Empty) return 0;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyRoomClipPlanes(
                Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), dest);
        }

        /// <summary>
        /// <c>SCREEN</c> (TV) stamps for the room. Clears
        /// <paramref name="dest"/>. At most 4. Empty when the UUID is
        /// missing or the room has no television.
        /// </summary>
        public int CopyScreenStamps(Guid sceneRoomUuid, List<ScanScreenStamp> dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (sceneRoomUuid == Guid.Empty) return 0;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyScreenStamps(
                Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), dest);
        }

        /// <summary>
        /// Conservative world AABB of the captured room (outer walls including
        /// doorway <c>INVISIBLE_WALL_FACE</c>, floor, ceiling), padded by the
        /// same 50 cm outward expand as the clip planes. Occupancy inset is
        /// not applied. False when the UUID is missing.
        /// </summary>
        public bool CopyRoomWorldAabb(Guid sceneRoomUuid, out Vector3 min, out Vector3 max)
        {
            min = max = Vector3.zero;
            if (sceneRoomUuid == Guid.Empty) return false;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return false;
            return Query.CopyRoomWorldAabb(
                Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), out min, out max);
        }

        /// <summary>
        /// Sample the room shell (outer walls, floor, ceiling, furniture
        /// faces) into <paramref name="dest"/> for the GPU coverage march.
        /// Doorway <c>INVISIBLE_WALL_FACE</c> anchors get no cells; wall
        /// cells inside a <c>DOOR_FRAME</c> / <c>WINDOW_FRAME</c> rect are
        /// excluded from the denominator. Clears <paramref name="dest"/>.
        /// Returns the number of uploadable cells (0 when the room is missing).
        /// </summary>
        public int CopyShellCells(Guid sceneRoomUuid, ShellCellSet dest)
        {
            if (dest == null) return 0;
            dest.Clear();
            if (sceneRoomUuid == Guid.Empty) return 0;
            EnsureMruk();
            if (_mruk == null || _mruk.Rooms == null) return 0;
            return Query.CopyShellCells(Query.FindByUuid(_mruk.Rooms, sceneRoomUuid), dest);
        }

        // ─────────────────────────────────────────────────────────────
        //  Scene Object Registry population
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Captures all MRUK anchors as SceneObjects and adds them to the registry.
        /// Each anchor gets a unique ID, full label, pose, bounds, and surface type.
        /// </summary>
        public void PopulateRegistry(SceneObjectRegistry registry)
        {
            if (registry == null) return;
            EnsureRoom();
            if (_room == null || _room.Anchors == null)
            {
                Logger.Warning("[RoomUnderstanding] PopulateRegistry — no MRUK room available");
                return;
            }

            int added = 0;
            int skipped = 0;
            for (int a = 0; a < _room.Anchors.Count; a++)
            {
                var anchor = _room.Anchors[a];
                string label = GetAnchorLabel(anchor);
                if (label == null)
                {
                    skipped++;
                    continue;
                }

                var t = anchor.transform;
                var surfType = ClassifyAnchor(anchor);

                var size = Vector3.one;
                var rot = t.rotation;
                bool hasVolume = anchor.VolumeBounds.HasValue;
                bool hasPlane = anchor.PlaneRect.HasValue;
                if (hasVolume)
                    size = anchor.VolumeBounds.Value.size;
                else if (hasPlane)
                {
                    var r = anchor.PlaneRect.Value;
                    size = new Vector3(r.width, r.height, 0.05f);
                }

                // anchor.transform.position is at the TOP face for volumes;
                // GetAnchorCenter() returns the true geometric center.
                var worldCenter = anchor.GetAnchorCenter();

                // Floor/ceiling anchors have an arbitrary in-plane yaw.
                // Derive the room's horizontal orientation from a wall anchor,
                // then project the PlaneRect corners into that wall-aligned frame
                // so the bounding box aligns with the physical room layout.
                if ((surfType == SurfaceType.Floor || surfType == SurfaceType.Ceiling) && hasPlane)
                {
                    var wallRight = FindWallHorizontalRight();
                    var wallPerp  = Vector3.Cross(Vector3.up, wallRight).normalized;

                    // Rotation: local X→wallRight, local Y→wallPerp, local Z→up (thin)
                    rot = Quaternion.LookRotation(Vector3.up, wallPerp);

                    // Project the anchor's PlaneRect corners into wall-aligned frame
                    var pr = anchor.PlaneRect.Value;
                    var hw = pr.width  * 0.5f;
                    var hh = pr.height * 0.5f;
                    var c0 = t.TransformPoint(new Vector3(-hw, -hh, 0));
                    var c1 = t.TransformPoint(new Vector3( hw, -hh, 0));
                    var c2 = t.TransformPoint(new Vector3( hw,  hh, 0));
                    var c3 = t.TransformPoint(new Vector3(-hw,  hh, 0));

                    float minW = float.MaxValue, maxW = float.MinValue;
                    float minD = float.MaxValue, maxD = float.MinValue;
                    foreach (var corner in new[] { c0, c1, c2, c3 })
                    {
                        var offset = corner - worldCenter;
                        float projW = Vector3.Dot(offset, wallRight);
                        float projD = Vector3.Dot(offset, wallPerp);
                        if (projW < minW) minW = projW; if (projW > maxW) maxW = projW;
                        if (projD < minD) minD = projD; if (projD > maxD) maxD = projD;
                    }

                    size = new Vector3(maxW - minW, maxD - minD, 0.05f);

                    // Re-center to the centroid of the wall-aligned bounding box
                    float offW = (minW + maxW) * 0.5f;
                    float offD = (minD + maxD) * 0.5f;
                    worldCenter += wallRight * offW + wallPerp * offD;
                }

                registry.Add(new SceneObject
                {
                    id = $"mruk_{a}_{label}",
                    label = label,
                    source = SceneObjectSource.MRUK,
                    surfaceType = surfType,
                    confidence = 1f,
                    position = worldCenter,
                    rotation = rot,
                    size = size,
                    mrukLabel = anchor.Label.ToString(),
                    anchorUuid = anchor.Anchor != null ? anchor.Anchor.Uuid.ToString() : ""
                });
                added++;

                Logger.Info($"[RoomUnderstanding] Anchor[{a}]: label={label}, " +
                            $"rawLabel={anchor.Label}, vol={hasVolume}, plane={hasPlane}, " +
                            $"size={size}, rot={rot.eulerAngles}, center={worldCenter}, anchorPos={t.position}");
            }
            Logger.Info($"[RoomUnderstanding] Populated {added} MRUK objects " +
                        $"(from {_room.Anchors.Count} anchors, {skipped} skipped)");
        }

        /// <summary>
        /// Returns a horizontal "right" direction along the first wall anchor found
        /// in the current room, giving us a reliable room-aligned axis.
        /// Falls back to world-right if no wall exists.
        /// </summary>
        private Vector3 FindWallHorizontalRight()
        {
            if (_room != null)
            {
                foreach (var a in _room.Anchors)
                {
                    if (!a.HasAnyLabel(WallLabels)) continue;
                    var right = a.transform.right;
                    right.y = 0;
                    if (right.sqrMagnitude > 0.001f)
                        return right.normalized;
                }
            }
            return Vector3.right;
        }

        private static string GetAnchorLabel(MRUKAnchor anchor)
        {
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.GLOBAL_MESH)) return null;
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR)) return "floor";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING)) return "ceiling";
            if (anchor.HasAnyLabel(WallLabels)) return "wall";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.DOOR_FRAME)) return "door";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.WINDOW_FRAME)) return "window";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.TABLE)) return "table";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.COUCH)) return "couch";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.BED)) return "bed";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.LAMP)) return "lamp";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.STORAGE)) return "storage";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.SCREEN)) return "screen";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.PLANT)) return "plant";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.WALL_ART)) return "wall_art";
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.OTHER)) return "other";
            return anchor.Label.ToString().ToLowerInvariant();
        }

        // ─────────────────────────────────────────────────────────────
        //  Internals
        // ─────────────────────────────────────────────────────────────

        void EnsureMruk()
        {
            if (_mruk == null)
                _mruk = FindAnyObjectByType<MRUK>();
        }

        /// <summary>
        /// Cached classification room is the volume that contains the
        /// headset. <c>GetCurrentRoom()</c> is last/first after you leave
        /// and must not be used. Hallway / no rooms → <c>_room</c> is null
        /// (do not fall back to <c>Rooms[0]</c>).
        /// </summary>
        private void EnsureRoom()
        {
            EnsureMruk();
            MRUKRoom next = null;
            if (_mruk != null && _mruk.Rooms != null)
                next = Query.FindContaining(_mruk.Rooms, Query.HeadsetWorldPosition());

            if (next != _room)
            {
                if (_room != null)
                    _room.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);
                _room = next;
                if (_room != null)
                    _room.AnchorCreatedEvent.AddListener(OnAnchorCreated);
            }

            SubscribeToRoomEvents();
        }

        private void SubscribeToRoomEvents()
        {
            if (_subscribedToRoomEvents) return;
            if (_mruk == null) return;

            _mruk.RoomCreatedEvent.AddListener(OnRoomCreatedOrUpdated);
            _mruk.RoomUpdatedEvent.AddListener(OnRoomCreatedOrUpdated);
            if (_mruk.Rooms != null)
            {
                for (int i = 0; i < _mruk.Rooms.Count; i++)
                {
                    var r = _mruk.Rooms[i];
                    if (r == null) continue;
                    r.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);
                    r.AnchorCreatedEvent.AddListener(OnAnchorCreated);
                }
            }
            _subscribedToRoomEvents = true;
        }

        private void UnsubscribeFromRoomEvents()
        {
            if (!_subscribedToRoomEvents) return;

            if (_mruk != null)
            {
                _mruk.RoomCreatedEvent.RemoveListener(OnRoomCreatedOrUpdated);
                _mruk.RoomUpdatedEvent.RemoveListener(OnRoomCreatedOrUpdated);
                if (_mruk.Rooms != null)
                {
                    for (int i = 0; i < _mruk.Rooms.Count; i++)
                    {
                        var r = _mruk.Rooms[i];
                        if (r != null)
                            r.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);
                    }
                }
            }

            if (_room != null)
                _room.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);

            _subscribedToRoomEvents = false;
        }

        private void OnRoomCreatedOrUpdated(MRUKRoom room)
        {
            if (room != null)
            {
                room.AnchorCreatedEvent.RemoveListener(OnAnchorCreated);
                room.AnchorCreatedEvent.AddListener(OnAnchorCreated);
            }
            EnsureRoom();
            Logger.Info($"[RoomUnderstanding] Room created/updated — " +
                        $"headset room anchors={_room?.Anchors?.Count ?? 0} " +
                        $"(event room={room?.Anchors?.Count ?? 0})");
            AnchorsChanged?.Invoke();
        }

        private void OnAnchorCreated(MRUKAnchor anchor)
        {
            Logger.Info($"[RoomUnderstanding] Anchor created: {anchor.Label}");
            AnchorsChanged?.Invoke();
        }

        /// <summary>Re-query MRUK for the current room (e.g., after scene reload).</summary>
        public void RefreshRoom()
        {
            UnsubscribeFromRoomEvents();
            _room = null;
            EnsureRoom();
        }

        private void OnDestroy()
        {
            UnsubscribeFromRoomEvents();
            if (Instance == this) Instance = null;
        }

        private void ClassifyFromMRUK(Vector3[] verts, SurfaceType[] result)
        {
            var anchors = _room.Anchors;
            for (int v = 0; v < verts.Length; v++)
            {
                MRUKAnchor best = null;
                float bestDist = float.MaxValue;
                for (int a = 0; a < anchors.Count; a++)
                {
                    float d = Vector3.SqrMagnitude(anchors[a].transform.position - verts[v]);
                    if (d < bestDist) { bestDist = d; best = anchors[a]; }
                }
                result[v] = best != null ? ClassifyAnchor(best) : SurfaceType.Unknown;
            }
        }

        private static void ClassifyFromNormals(Vector3[] normals, SurfaceType[] result)
        {
            for (int i = 0; i < normals.Length; i++)
            {
                float ny = normals[i].y;
                if (ny > 0.7f) result[i] = SurfaceType.Floor;
                else if (ny < -0.7f) result[i] = SurfaceType.Ceiling;
                else result[i] = SurfaceType.Wall;
            }
        }

        private const MRUKAnchor.SceneLabels WallLabels =
            MRUKAnchor.SceneLabels.WALL_FACE |
            MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE |
            MRUKAnchor.SceneLabels.INNER_WALL_FACE;

        private const MRUKAnchor.SceneLabels FurnitureLabels =
            MRUKAnchor.SceneLabels.TABLE |
            MRUKAnchor.SceneLabels.COUCH |
            MRUKAnchor.SceneLabels.BED |
            MRUKAnchor.SceneLabels.LAMP |
            MRUKAnchor.SceneLabels.STORAGE |
            MRUKAnchor.SceneLabels.SCREEN |
            MRUKAnchor.SceneLabels.PLANT |
            MRUKAnchor.SceneLabels.WALL_ART |
            MRUKAnchor.SceneLabels.OTHER;

        private const MRUKAnchor.SceneLabels StructureLabels =
            MRUKAnchor.SceneLabels.DOOR_FRAME |
            MRUKAnchor.SceneLabels.WINDOW_FRAME;

        private static SurfaceType ClassifyAnchor(MRUKAnchor anchor)
        {
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.GLOBAL_MESH))
                return SurfaceType.Unknown;
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR))
                return SurfaceType.Floor;
            if (anchor.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING))
                return SurfaceType.Ceiling;
            if (anchor.HasAnyLabel(WallLabels))
                return SurfaceType.Wall;
            if (anchor.HasAnyLabel(StructureLabels))
                return SurfaceType.Wall;
            if (anchor.HasAnyLabel(FurnitureLabels))
                return SurfaceType.Furniture;
            return SurfaceType.Furniture;
        }

        private static bool IsWall(MRUKAnchor anchor) => anchor.HasAnyLabel(WallLabels);
        private static bool IsFurniture(MRUKAnchor anchor) => anchor.HasAnyLabel(FurnitureLabels);

        /// <summary>
        /// Shared wall-plane occupancy. Native <c>IsPositionInRoom</c> is the
        /// floor outline; Horizon Space Setup uses the walls.
        /// </summary>
        internal static class Query
        {
            const float OuterWallInsetMetres = 0.08f;
            /// <summary>
            /// TSDF clip hull is the MRUK outer walls / floor / ceiling
            /// pushed this far <b>outward</b> once, then confined hard.
            /// Covers a false ceiling above the Scene API plane, wall
            /// plaster, and Quest depth noise. Occupancy inset is unchanged.
            /// </summary>
            const float TsdfClipExpandMetres = 0.50f;

            const MRUKAnchor.SceneLabels OuterWallLabels =
                MRUKAnchor.SceneLabels.WALL_FACE
                | MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE;

            const MRUKAnchor.SceneLabels WallArtLabels =
                MRUKAnchor.SceneLabels.WALL_ART;

            const float MinVerticalPlaneMetres = 0.2f;

            // Hosts poll occupancy every frame (hide a look when the headset
            // leaves the room), so the rig lookup is cached rather than a
            // scene search per call. Re-resolved if the anchor is destroyed.
            static Transform _centerEye;

            internal static Vector3 HeadsetWorldPosition()
            {
                if (_centerEye == null)
                {
                    var rig = UnityEngine.Object.FindAnyObjectByType<OVRCameraRig>(
                        FindObjectsInactive.Include);
                    if (rig != null)
                        _centerEye = rig.centerEyeAnchor;
                }
                if (_centerEye != null)
                    return _centerEye.position;
                var cam = Camera.main;
                return cam != null ? cam.transform.position : Vector3.zero;
            }

            internal static Guid RoomUuid(MRUKRoom room)
            {
                if (room == null || room.Anchor == OVRAnchor.Null)
                    return Guid.Empty;
                return room.Anchor.Uuid;
            }

            internal static bool Contains(MRUKRoom room, Vector3 worldPos)
            {
                if (room == null) return false;
                if (!room.IsPositionInRoom(worldPos, testVerticalBounds: true))
                    return false;
                return InsideOuterWalls(room, worldPos);
            }

            internal static MRUKRoom FindContaining(IList<MRUKRoom> rooms, Vector3 worldPos)
            {
                if (rooms == null) return null;
                for (int i = 0; i < rooms.Count; i++)
                {
                    var room = rooms[i];
                    if (Contains(room, worldPos))
                        return room;
                }
                return null;
            }

            internal static MRUKRoom FindByUuid(IList<MRUKRoom> rooms, Guid uuid)
            {
                if (rooms == null || uuid == Guid.Empty) return null;
                for (int i = 0; i < rooms.Count; i++)
                {
                    var room = rooms[i];
                    if (RoomUuid(room) == uuid)
                        return room;
                }
                return null;
            }

            internal static int CopyWallFacesAt(
                IList<MRUKRoom> rooms,
                Vector3 worldPos,
                List<SceneWallFace> dst,
                SceneFaceKind kind)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (rooms == null || kind == SceneFaceKind.None) return 0;
                for (int i = 0; i < rooms.Count; i++)
                {
                    var room = rooms[i];
                    if (!Contains(room, worldPos)) continue;
                    AppendWallFaces(room, dst, kind);
                }
                return dst.Count;
            }

            /// <summary>
            /// Names of the Scene API labels (<see cref="MRUKAnchor.SceneLabels"/>) carried by the anchors of every
            /// room that contains <paramref name="worldPos"/>, each once. Clears <paramref name="dst"/>.
            /// </summary>
            internal static int CopyLabelsAt(IList<MRUKRoom> rooms, Vector3 worldPos, List<string> dst)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (rooms == null) return 0;
                for (int i = 0; i < rooms.Count; i++)
                {
                    var room = rooms[i];
                    if (room == null || room.Anchors == null || !Contains(room, worldPos)) continue;
                    for (int a = 0; a < room.Anchors.Count; a++)
                    {
                        var anchor = room.Anchors[a];
                        if (anchor != null) AppendLabelNames(anchor.Label, dst);
                    }
                }
                return dst.Count;
            }

            static readonly MRUKAnchor.SceneLabels[] SingleLabels = SingleLabelValues();

            static MRUKAnchor.SceneLabels[] SingleLabelValues()
            {
                var all = (MRUKAnchor.SceneLabels[])Enum.GetValues(typeof(MRUKAnchor.SceneLabels));
                var singles = new List<MRUKAnchor.SceneLabels>(all.Length);
                foreach (var v in all)
                {
                    ulong bits = unchecked((uint)Convert.ToInt32(v)); // a flag at bit 31 is negative as an int
                    if (bits != 0 && (bits & (bits - 1)) == 0) singles.Add(v);
                }
                return singles.ToArray();
            }

            static void AppendLabelNames(MRUKAnchor.SceneLabels labels, List<string> dst)
            {
                for (int i = 0; i < SingleLabels.Length; i++)
                {
                    var flag = SingleLabels[i];
                    if ((labels & flag) == 0) continue;
                    string name = flag.ToString();
                    if (!dst.Contains(name)) dst.Add(name);
                }
            }

            internal static int CopyWallFaces(
                MRUKRoom room, List<SceneWallFace> dst, SceneFaceKind kind)
            {
                if (dst == null) return 0;
                dst.Clear();
                AppendWallFaces(room, dst, kind);
                return dst.Count;
            }

            static void AppendWallFaces(MRUKRoom room, List<SceneWallFace> dst, SceneFaceKind kind)
            {
                if (room == null || room.Anchors == null) return;
                float floorY = FloorY(room);
                for (int i = 0; i < room.Anchors.Count; i++)
                    TryAddVerticalPlane(room, room.Anchors[i], floorY, dst, kind);
            }

            static bool MatchesKind(MRUKAnchor a, SceneFaceKind kind)
            {
                bool screen = a.HasAnyLabel(MRUKAnchor.SceneLabels.SCREEN);
                if (screen)
                    return (kind & SceneFaceKind.Screen) != 0;
                bool wall = a.HasAnyLabel(MRUKAnchor.SceneLabels.WALL_FACE)
                    && !a.HasAnyLabel(MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE)
                    && !a.HasAnyLabel(MRUKAnchor.SceneLabels.INNER_WALL_FACE);
                return wall && (kind & SceneFaceKind.Wall) != 0;
            }

            static void TryAddVerticalPlane(
                MRUKRoom room,
                MRUKAnchor a,
                float floorY,
                List<SceneWallFace> dst,
                SceneFaceKind kind)
            {
                if (a == null || !MatchesKind(a, kind)) return;

                Vector3 inward = Inward(room, a);
                float upDot = Vector3.Dot(inward, Vector3.up);
                if (upDot > 0.7f || upDot < -0.7f)
                    return;

                if (!TryPlaneSize(a, out Vector3 center, out float width, out float height))
                    return;

                if (width < MinVerticalPlaneMetres || height < MinVerticalPlaneMetres)
                    return;

                bool avoid = a.HasAnyLabel(WallArtLabels);
                dst.Add(new SceneWallFace(
                    center, inward, width, height, floorY, avoid));
            }

            static bool TryPlaneSize(
                MRUKAnchor a, out Vector3 center, out float width, out float height)
            {
                center = default;
                width = 0f;
                height = 0f;

                if (a.PlaneRect.HasValue)
                {
                    var rect = a.PlaneRect.Value;
                    Vector3 worldX = a.transform.TransformVector(new Vector3(rect.width, 0f, 0f));
                    Vector3 worldY = a.transform.TransformVector(new Vector3(0f, rect.height, 0f));
                    float horizX = Vector3.ProjectOnPlane(worldX, Vector3.up).magnitude;
                    float horizY = Vector3.ProjectOnPlane(worldY, Vector3.up).magnitude;
                    width = Mathf.Max(horizX, horizY);
                    height = Mathf.Max(
                        Mathf.Abs(Vector3.Dot(worldX, Vector3.up)),
                        Mathf.Abs(Vector3.Dot(worldY, Vector3.up)));
                    center = a.transform.TransformPoint(rect.center);
                    return true;
                }

                if (!a.VolumeBounds.HasValue) return false;

                var size = a.VolumeBounds.Value.size;
                Vector3 wx = a.transform.TransformVector(new Vector3(size.x, 0f, 0f));
                Vector3 wy = a.transform.TransformVector(new Vector3(0f, size.y, 0f));
                Vector3 wz = a.transform.TransformVector(new Vector3(0f, 0f, size.z));
                float hx = Vector3.ProjectOnPlane(wx, Vector3.up).magnitude;
                float hy = Vector3.ProjectOnPlane(wy, Vector3.up).magnitude;
                float hz = Vector3.ProjectOnPlane(wz, Vector3.up).magnitude;
                width = Mathf.Max(hx, Mathf.Max(hy, hz));
                height = Mathf.Max(
                    Mathf.Abs(Vector3.Dot(wx, Vector3.up)),
                    Mathf.Max(
                        Mathf.Abs(Vector3.Dot(wy, Vector3.up)),
                        Mathf.Abs(Vector3.Dot(wz, Vector3.up))));
                center = a.GetAnchorCenter();
                return true;
            }

            static float FloorY(MRUKRoom room)
            {
                if (room.FloorAnchors != null && room.FloorAnchors.Count > 0)
                {
                    var f = room.FloorAnchors[0];
                    if (f != null) return f.GetAnchorCenter().y;
                }

                return 0f;
            }

            static Vector3 Inward(MRUKRoom room, MRUKAnchor a)
            {
                Vector3 inward = room.GetFacingDirection(a);
                if (inward.sqrMagnitude < 1e-8f)
                    inward = a.transform.forward;
                inward.Normalize();
                return inward;
            }

            static bool InsideOuterWalls(MRUKRoom room, Vector3 worldPos)
            {
                var anchors = room.Anchors;
                if (anchors == null || anchors.Count == 0)
                    return true;

                for (int i = 0; i < anchors.Count; i++)
                {
                    var a = anchors[i];
                    if (a == null) continue;
                    if (!a.HasAnyLabel(OuterWallLabels)) continue;

                    if (Vector3.Dot(worldPos - a.transform.position, Inward(room, a))
                        < OuterWallInsetMetres)
                        return false;
                }

                return true;
            }

            const int MaxRoomClipPlanes = VolumeIntegrator.MaxRoomClipPlanes;
            const int MaxScreenStamps = VolumeIntegrator.MaxScreenStamps;

            static void AddExpandedPlane(List<Vector4> dst, Vector3 inwardUnit, Vector3 pointOnPlane)
            {
                float w = Vector3.Dot(pointOnPlane, inwardUnit) - TsdfClipExpandMetres;
                dst.Add(new Vector4(inwardUnit.x, inwardUnit.y, inwardUnit.z, w));
            }

            internal static int CopyRoomClipPlanes(MRUKRoom room, List<Vector4> dst)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (room == null || room.Anchors == null) return 0;

                for (int i = 0; i < room.Anchors.Count && dst.Count < MaxRoomClipPlanes; i++)
                {
                    var a = room.Anchors[i];
                    if (a == null) continue;
                    if (!a.HasAnyLabel(OuterWallLabels)) continue;

                    Vector3 n = Inward(room, a);
                    AddExpandedPlane(dst, n, a.transform.position);
                }

                if (dst.Count < MaxRoomClipPlanes)
                    AddExpandedPlane(dst, Vector3.up, new Vector3(0f, FloorY(room), 0f));

                if (dst.Count < MaxRoomClipPlanes
                    && room.CeilingAnchors != null
                    && room.CeilingAnchors.Count > 0
                    && room.CeilingAnchors[0] != null)
                {
                    float cy = room.CeilingAnchors[0].GetAnchorCenter().y;
                    AddExpandedPlane(dst, Vector3.down, new Vector3(0f, cy, 0f));
                }

                return dst.Count;
            }

            internal static int CopyScreenStamps(MRUKRoom room, List<ScanScreenStamp> dst)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (room == null) return 0;

                var faces = new List<SceneWallFace>(4);
                CopyWallFaces(room, faces, SceneFaceKind.Screen);
                for (int i = 0; i < faces.Count && dst.Count < MaxScreenStamps; i++)
                    dst.Add(ScanScreenStamp.FromFace(faces[i]));

                return dst.Count;
            }

            internal static bool CopyRoomWorldAabb(
                MRUKRoom room, out Vector3 min, out Vector3 max)
            {
                min = max = Vector3.zero;
                if (room == null) return false;

                bool any = false;
                Bounds b = default;
                if (room.Anchors != null)
                {
                    for (int i = 0; i < room.Anchors.Count; i++)
                    {
                        var a = room.Anchors[i];
                        if (a == null) continue;
                        if (!a.HasAnyLabel(OuterWallLabels)
                            && !a.HasAnyLabel(MRUKAnchor.SceneLabels.FLOOR)
                            && !a.HasAnyLabel(MRUKAnchor.SceneLabels.CEILING))
                            continue;
                        EncapsulateAnchor(a, ref b, ref any);
                    }
                }

                if (!any) return false;
                Vector3 pad = Vector3.one * TsdfClipExpandMetres;
                min = b.min - pad;
                max = b.max + pad;
                return true;
            }

            static void EncapsulateAnchor(MRUKAnchor a, ref Bounds b, ref bool any)
            {
                if (a.PlaneRect.HasValue)
                {
                    var r = a.PlaneRect.Value;
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMin, r.yMin, 0f)), ref b, ref any);
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMax, r.yMin, 0f)), ref b, ref any);
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMin, r.yMax, 0f)), ref b, ref any);
                    EncapsulatePoint(a.transform.TransformPoint(new Vector3(r.xMax, r.yMax, 0f)), ref b, ref any);
                    return;
                }

                if (!a.VolumeBounds.HasValue) return;
                var vb = a.VolumeBounds.Value;
                Vector3 c = vb.center;
                Vector3 e = vb.extents;
                for (int x = -1; x <= 1; x += 2)
                for (int y = -1; y <= 1; y += 2)
                for (int z = -1; z <= 1; z += 2)
                    EncapsulatePoint(
                        a.transform.TransformPoint(c + Vector3.Scale(e, new Vector3(x, y, z))),
                        ref b, ref any);
            }

            static void EncapsulatePoint(Vector3 p, ref Bounds b, ref bool any)
            {
                if (!any)
                {
                    b = new Bounds(p, Vector3.zero);
                    any = true;
                }
                else
                {
                    b.Encapsulate(p);
                }
            }

            // ── Shell cells for the coverage march ──────────────────────

            /// <summary>Shell cells target ~14k so a 16k cap leaves headroom.</summary>
            const float ShellTargetCells = 14000f;
            const float ShellCellMin = 0.10f;
            const float ShellCellMax = 0.20f;
            /// <summary>
            /// Plane cells march from the 50 cm clip expand outside the plane
            /// to a per-kind depth inside. Walls go 1.0 m so the front of a
            /// desk or wardrobe standing against the wall still covers the
            /// wall behind it; the ceiling 0.8 m for the top of tall storage;
            /// the floor 0.7 m so a bed or couch covers the floor under it
            /// while a 0.75 m table does not.
            /// </summary>
            const float ShellMarchStart = -TsdfClipExpandMetres;
            const float WallMarchLength = TsdfClipExpandMetres + 1.00f;
            const float CeilingMarchLength = TsdfClipExpandMetres + 0.80f;
            const float FloorMarchLength = TsdfClipExpandMetres + 0.70f;
            /// <summary>
            /// Furniture faces march from just outside the face through the
            /// whole volume box. A scene box bounds the object loosely (a
            /// couch box top is the backrest, a bed box top the headboard, a
            /// table box side is mostly air between legs), so the real surface
            /// may sit anywhere along that depth — and where there is nothing
            /// at all the march reports observed-empty and the cell leaves the
            /// denominator instead of reading as a hole.
            /// </summary>
            const float FurnitureMarchOutside = 0.25f;
            /// <summary>Top faces stop short of the box bottom so the floor band is never the hit.</summary>
            const float FurnitureBottomSkip = 0.10f;
            const float OpeningMargin = 0.05f;
            const float OpeningDepth = 0.15f;
            /// <summary>
            /// A side face within this distance of a wall is skipped: the gap
            /// behind a couch or bed cannot be scanned and is not a leak.
            /// </summary>
            const float FurnitureWallSkip = 0.60f;
            const float FurnitureFloorSkip = 0.10f;

            const MRUKAnchor.SceneLabels ShellFurnitureLabels =
                MRUKAnchor.SceneLabels.TABLE
                | MRUKAnchor.SceneLabels.COUCH
                | MRUKAnchor.SceneLabels.BED
                | MRUKAnchor.SceneLabels.STORAGE;

            const MRUKAnchor.SceneLabels OpeningLabels =
                MRUKAnchor.SceneLabels.DOOR_FRAME
                | MRUKAnchor.SceneLabels.WINDOW_FRAME;

            static readonly List<MRUKAnchor> s_openings = new(8);
            static readonly List<MRUKAnchor> s_walls = new(16);

            static bool IsShellWall(MRUKAnchor a)
                => a.HasAnyLabel(MRUKAnchor.SceneLabels.WALL_FACE)
                   && !a.HasAnyLabel(MRUKAnchor.SceneLabels.INVISIBLE_WALL_FACE)
                   && !a.HasAnyLabel(MRUKAnchor.SceneLabels.INNER_WALL_FACE)
                   && a.PlaneRect.HasValue;

            internal static int CopyShellCells(MRUKRoom room, ShellCellSet dst)
            {
                if (dst == null) return 0;
                dst.Clear();
                if (room == null || room.Anchors == null) return 0;

                s_openings.Clear();
                s_walls.Clear();
                float area = 0f;
                for (int i = 0; i < room.Anchors.Count; i++)
                {
                    var a = room.Anchors[i];
                    if (a == null) continue;
                    if (a.HasAnyLabel(OpeningLabels) && a.PlaneRect.HasValue)
                        s_openings.Add(a);
                    if (a.HasAnyLabel(OuterWallLabels))
                        s_walls.Add(a);
                    if (IsShellWall(a))
                    {
                        var r = a.PlaneRect.Value;
                        area += r.width * r.height;
                    }
                }
                // High Fidelity scenes can carry several floor / ceiling planes.
                var floors = room.FloorAnchors;
                var ceilings = room.CeilingAnchors;
                area += PlaneListArea(floors) + PlaneListArea(ceilings);

                float cell = Mathf.Clamp(Mathf.Sqrt(area / ShellTargetCells), ShellCellMin, ShellCellMax);
                dst.CellSize = cell;
                float floorY = FloorY(room);

                for (int i = 0; i < s_walls.Count; i++)
                {
                    var a = s_walls[i];
                    if (!IsShellWall(a)) continue;
                    AddWallCells(room, a, cell, dst);
                }
                if (floors != null)
                    for (int i = 0; i < floors.Count; i++)
                        if (floors[i] != null)
                            AddPolygonCells(floors[i], Vector3.up, ShellSurfaceKind.Floor, cell, dst);
                if (ceilings != null)
                    for (int i = 0; i < ceilings.Count; i++)
                        if (ceilings[i] != null)
                            AddPolygonCells(ceilings[i], Vector3.down, ShellSurfaceKind.Ceiling, cell, dst);

                for (int i = 0; i < room.Anchors.Count; i++)
                {
                    var a = room.Anchors[i];
                    if (a == null || !a.HasAnyLabel(ShellFurnitureLabels) || !a.VolumeBounds.HasValue)
                        continue;
                    AddFurnitureCells(room, a, cell, floorY, dst);
                }

                return dst.UploadCount;
            }

            static float PlaneListArea(List<MRUKAnchor> anchors)
            {
                if (anchors == null) return 0f;
                float area = 0f;
                for (int i = 0; i < anchors.Count; i++)
                {
                    var a = anchors[i];
                    if (a == null || !a.PlaneRect.HasValue) continue;
                    area += a.PlaneRect.Value.width * a.PlaneRect.Value.height;
                }
                return area;
            }

            static void AddWallCells(MRUKRoom room, MRUKAnchor a, float cell, ShellCellSet dst)
            {
                var r = a.PlaneRect.Value;
                int nu = Mathf.CeilToInt(r.width / cell);
                int nv = Mathf.CeilToInt(r.height / cell);
                var t = a.transform;
                Vector3 inward = Inward(room, a);
                int s = dst.BeginSurface(ShellSurfaceKind.Wall, t.right, t.up, inward, nu, nv);
                if (s < 0) return;

                for (int v = 0; v < nv; v++)
                for (int u = 0; u < nu; u++)
                {
                    float lx = r.xMin + (u + 0.5f) * cell;
                    float ly = r.yMin + (v + 0.5f) * cell;
                    Vector3 p = t.TransformPoint(new Vector3(lx, ly, 0f));
                    bool excluded = InsideOpening(p);
                    dst.AddCell(s, u, v, p, ShellMarchStart, WallMarchLength, excluded);
                }
            }

            static void AddPolygonCells(MRUKAnchor a, Vector3 normal, ShellSurfaceKind kind, float cell, ShellCellSet dst)
            {
                if (!a.PlaneRect.HasValue) return;
                var r = a.PlaneRect.Value;
                var poly = a.PlaneBoundary2D;
                int nu = Mathf.CeilToInt(r.width / cell);
                int nv = Mathf.CeilToInt(r.height / cell);
                var t = a.transform;
                int s = dst.BeginSurface(kind, t.right, t.up, normal, nu, nv);
                if (s < 0) return;

                float length = kind == ShellSurfaceKind.Floor ? FloorMarchLength : CeilingMarchLength;
                bool usePoly = poly != null && poly.Count >= 3;
                for (int v = 0; v < nv; v++)
                for (int u = 0; u < nu; u++)
                {
                    var l = new Vector2(r.xMin + (u + 0.5f) * cell, r.yMin + (v + 0.5f) * cell);
                    bool inside = !usePoly || PointInPolygon(poly, l);
                    Vector3 p = t.TransformPoint(new Vector3(l.x, l.y, 0f));
                    dst.AddCell(s, u, v, p, ShellMarchStart, length, !inside);
                }
            }

            static void AddFurnitureCells(MRUKRoom room, MRUKAnchor a, float cell, float floorY, ShellCellSet dst)
            {
                var vb = a.VolumeBounds.Value;
                Vector3 c = vb.center;
                Vector3 e = vb.extents;
                var t = a.transform;

                for (int axis = 0; axis < 3; axis++)
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    Vector3 ln = Vector3.zero;
                    ln[axis] = sign;
                    Vector3 wn = t.TransformDirection(ln).normalized;
                    float upDot = Vector3.Dot(wn, Vector3.up);
                    if (upDot < -0.7f) continue; // underside

                    int ai = (axis + 1) % 3;
                    int bi = (axis + 2) % 3;
                    float ea = e[ai];
                    float eb = e[bi];
                    if (ea * 2f < cell || eb * 2f < cell) continue;

                    Vector3 faceCenterLocal = c + ln * e[axis];
                    Vector3 faceCenter = t.TransformPoint(faceCenterLocal);
                    bool side = Mathf.Abs(upDot) < 0.7f;
                    if (side)
                    {
                        if (FaceAgainstWall(room, faceCenter, wn)) continue;
                        if (faceCenter.y - floorY < FurnitureFloorSkip) continue;
                    }

                    // The march runs inward (against the face normal): from
                    // just outside the face to the far side of the box, or
                    // to just above the box bottom for the top face.
                    float depth = e[axis] * 2f;
                    if (!side) depth = Mathf.Max(depth - FurnitureBottomSkip, cell);
                    float marchStart = -FurnitureMarchOutside;
                    float marchLength = FurnitureMarchOutside + depth;

                    Vector3 la = Vector3.zero; la[ai] = 1f;
                    Vector3 lb = Vector3.zero; lb[bi] = 1f;
                    int nu = Mathf.CeilToInt(ea * 2f / cell);
                    int nv = Mathf.CeilToInt(eb * 2f / cell);
                    int s = dst.BeginSurface(ShellSurfaceKind.Furniture,
                        t.TransformDirection(la).normalized, t.TransformDirection(lb).normalized, -wn, nu, nv);
                    if (s < 0) return;

                    for (int v = 0; v < nv; v++)
                    for (int u = 0; u < nu; u++)
                    {
                        Vector3 lp = faceCenterLocal
                            + la * (-ea + (u + 0.5f) * cell)
                            + lb * (-eb + (v + 0.5f) * cell);
                        dst.AddCell(s, u, v, t.TransformPoint(lp), marchStart, marchLength, false);
                    }
                }
            }

            static bool FaceAgainstWall(MRUKRoom room, Vector3 faceCenter, Vector3 faceNormal)
            {
                for (int i = 0; i < s_walls.Count; i++)
                {
                    var w = s_walls[i];
                    Vector3 inward = Inward(room, w);
                    if (Vector3.Dot(faceNormal, inward) > -0.7f) continue;
                    float d = Vector3.Dot(faceCenter - w.transform.position, inward);
                    if (d >= -OpeningMargin && d < FurnitureWallSkip)
                        return true;
                }
                return false;
            }

            static bool InsideOpening(Vector3 worldPos)
            {
                for (int i = 0; i < s_openings.Count; i++)
                {
                    var o = s_openings[i];
                    var r = o.PlaneRect.Value;
                    Vector3 l = o.transform.InverseTransformPoint(worldPos);
                    if (Mathf.Abs(l.z) > OpeningDepth) continue;
                    if (l.x < r.xMin - OpeningMargin || l.x > r.xMax + OpeningMargin) continue;
                    if (l.y < r.yMin - OpeningMargin || l.y > r.yMax + OpeningMargin) continue;
                    return true;
                }
                return false;
            }

            static bool PointInPolygon(List<Vector2> poly, Vector2 p)
            {
                bool inside = false;
                int n = poly.Count;
                for (int i = 0, j = n - 1; i < n; j = i++)
                {
                    Vector2 a = poly[i];
                    Vector2 b = poly[j];
                    if ((a.y > p.y) != (b.y > p.y)
                        && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
                        inside = !inside;
                }
                return inside;
            }
        }
    }
}
