Shader "Genesis/ScanMeshVertexColor"
{
    Properties { }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        Pass
        {
            Name "VertexColorUnlit"
            Tags { "LightMode"="SRPDefaultUnlit" }
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "ScanMeshBirth.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 prevPos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
                uint   _pad;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;

            half4 UnpackColor(uint packed)
            {
                return half4(
                    (packed        & 0xFF) / 255.0h,
                    ((packed >> 8) & 0xFF) / 255.0h,
                    ((packed >> 16)& 0xFF) / 255.0h,
                    ((packed >> 24)& 0xFF) / 255.0h);
            }

            // ── Triplanar persistent textures ──
            TEXTURE2D(_RSTriXZ);  SAMPLER(sampler_RSTriXZ);
            TEXTURE2D(_RSTriXY);  SAMPLER(sampler_RSTriXY);
            TEXTURE2D(_RSTriYZ);  SAMPLER(sampler_RSTriYZ);
            TEXTURE2D(_RSTriDepthXZ);  SAMPLER(sampler_RSTriDepthXZ);
            TEXTURE2D(_RSTriDepthXY);  SAMPLER(sampler_RSTriDepthXY);
            TEXTURE2D(_RSTriDepthYZ);  SAMPLER(sampler_RSTriDepthYZ);
            float _RSTriAvailable;

            // ── TSDF volume (for freeze tint) ──
            TEXTURE3D(gsVolume);
            SAMPLER(sampler_gsVolume);
            float4 gsVoxCount;
            float gsVoxSize;
            // ── Analysis labels (R8: label / 5). 4 = free voxel on a leak face. ──
            TEXTURE3D(gsLabelVolume);
            SAMPLER(sampler_gsLabelVolume);

            // ── Globals set by RoomScanner ──
            float _RSNoFreezeTint;
            float _RSNormalFallback;
            float _RSWireframe;
            float _RSWireThickness;
            // 1 = tint the surface where the analysis found a leak face.
            float _RSShowHoles;
            // Freeze preview cones (RoomScanner.SetFreezePreview): A = apex xyz, cos(half-angle);
            // B = axis xyz, max metres (0 unbounded); C = colour rgb, strength (0 = off).
            float4 _RSPreviewCone0A, _RSPreviewCone0B, _RSPreviewCone0C;
            float4 _RSPreviewCone1A, _RSPreviewCone1B, _RSPreviewCone1C;

            #define DEPTH_TOLERANCE 0.015

            float3 WorldToVoxelUVW(float3 worldPos)
            {
                float3 local = worldPos / gsVoxSize + gsVoxCount.xyz / 2.0;
                return saturate(local / gsVoxCount.xyz);
            }

            float2 SignedTriUV(float2 baseUV, float normalComponent)
            {
                return float2(baseUV.x, normalComponent > 0 ? baseUV.y * 0.5 + 0.5 : baseUV.y * 0.5);
            }

            half3 SampleTriplanar(float3 worldPos, float3 normal)
            {
                float3 absN   = abs(normal);
                float3 blend  = absN / (absN.x + absN.y + absN.z + 0.001);
                float3 uvw    = WorldToVoxelUVW(worldPos);

                float2 uvXZ = SignedTriUV(uvw.xz, normal.y);
                float2 uvXY = SignedTriUV(uvw.xy, normal.z);
                float2 uvYZ = SignedTriUV(uvw.yz, normal.x);

                half4 colXZ = SAMPLE_TEXTURE2D(_RSTriXZ, sampler_RSTriXZ, uvXZ);
                half4 colXY = SAMPLE_TEXTURE2D(_RSTriXY, sampler_RSTriXY, uvXY);
                half4 colYZ = SAMPLE_TEXTURE2D(_RSTriYZ, sampler_RSTriYZ, uvYZ);

                float dXZ = SAMPLE_TEXTURE2D(_RSTriDepthXZ, sampler_RSTriDepthXZ, uvXZ).r;
                float dXY = SAMPLE_TEXTURE2D(_RSTriDepthXY, sampler_RSTriDepthXY, uvXY).r;
                float dYZ = SAMPLE_TEXTURE2D(_RSTriDepthYZ, sampler_RSTriDepthYZ, uvYZ).r;

                if (dXZ > 0.001 && abs(uvw.y - dXZ) > DEPTH_TOLERANCE) colXZ = half4(0, 0, 0, 0);
                if (dXY > 0.001 && abs(uvw.z - dXY) > DEPTH_TOLERANCE) colXY = half4(0, 0, 0, 0);
                if (dYZ > 0.001 && abs(uvw.x - dYZ) > DEPTH_TOLERANCE) colYZ = half4(0, 0, 0, 0);

                half3 rgb = colXZ.rgb * blend.y + colXY.rgb * blend.z + colYZ.rgb * blend.x;
                half totalAlpha = colXZ.a * blend.y + colXY.a * blend.z + colYZ.a * blend.x;

                return totalAlpha > 0.01 ? rgb : half3(-1, -1, -1);
            }

            half3 ApplyBirthFade(half3 color, half fade)
            {
                if (_RSBirthFadeSec < 0.001h)
                    return color;
                half3 gate = half3(0.35h, 0.78h, 0.95h);
                return lerp(color * gate, color, fade);
            }

            bool IsVoxelFrozen(float3 worldPos)
            {
                float3 uvw = WorldToVoxelUVW(worldPos);
                float2 tsdf = SAMPLE_TEXTURE3D_LOD(gsVolume, sampler_gsVolume, uvw, 0).rg;
                return tsdf.g < 0;
            }

            half3 ApplyFreezeTint(half3 color, float3 worldPos)
            {
                if (_RSNoFreezeTint < 0.5 && IsVoxelFrozen(worldPos))
                    color = lerp(color, half3(0.3, 0.5, 0.9), 0.25);
                return color;
            }

            // A freeze preview cone lights the surfaces a FreezeInView / UnfreezeInView with the same cone would
            // paint (the same test as gsInFreezeCone: past 5 cm of the apex, within the angle and the length): a
            // drifting fill and a brighter rim, so the edge of what will change is plain.
            half3 ApplyPreviewCone(half3 color, float3 p, float4 a, float4 b, float4 c)
            {
                if (c.a <= 0.001) return color;
                float3 d = p - a.xyz;
                float dist = length(d);
                if (dist < 0.05) return color;
                float along = dot(d, b.xyz);
                float cosAng = along / dist;
                if (cosAng < a.w) return color;
                if (b.w > 0.0 && along > b.w) return color;
                float t = saturate((cosAng - a.w) / max(1.0 - a.w, 1e-4));
                float rim = 1.0 - smoothstep(0.0, 0.14, t);
                float fill = 0.30 + 0.08 * sin(_Time.y * 4.0 - dist * 16.0);
                return lerp(color, c.rgb, saturate((fill + rim * 0.55) * c.a));
            }

            // The leak-marked voxel is the free one just in front of the
            // surface, so look one voxel out along the normal (and at the
            // surface itself) — whichever lands in it. Sampled per vertex,
            // not per fragment: two dependent 3-D fetches on a 16 MB volume
            // per stereo pixel cost frames; per vertex they are free.
            float NearLeak(float3 worldPos, float3 normal)
            {
                float a = SAMPLE_TEXTURE3D_LOD(gsLabelVolume, sampler_gsLabelVolume,
                    WorldToVoxelUVW(worldPos + normal * gsVoxSize), 0).r;
                float b = SAMPLE_TEXTURE3D_LOD(gsLabelVolume, sampler_gsLabelVolume,
                    WorldToVoxelUVW(worldPos), 0).r;
                return (round(a * 5.0) == 4.0 || round(b * 5.0) == 4.0) ? 1.0 : 0.0;
            }

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float4 color : COLOR;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float3 barycentric : TEXCOORD2;
                float hole : TEXCOORD3;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                uint idx = _SurfaceIndices[vertID];
                GPUVertex gv = _SurfaceVerts[idx];
                half4 unpacked = UnpackColor(gv.packedColor);
                float fade = ScanMeshBirthFade(unpacked.a, gv.voxelFlatIdx);
                float3 pos = ScanMeshPresentedPos(gv.pos, gv.prevPos, gv.norm, fade);

                OUT.positionWS  = pos;
                OUT.positionHCS = TransformWorldToHClip(pos);
                OUT.normalWS    = gv.norm;
                OUT.color       = half4(unpacked.rgb, fade);
                OUT.hole        = _RSShowHoles > 0.5 ? NearLeak(gv.pos, gv.norm) : 0.0;

                // Barycentric coords for wireframe: each triangle vertex gets one axis
                uint triVert = vertID % 3;
                OUT.barycentric = triVert == 0 ? float3(1, 0, 0)
                                : triVert == 1 ? float3(0, 1, 0)
                                :                float3(0, 0, 1);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float3 normal = normalize(IN.normalWS);

                // 1. Compute base color
                half3 baseColor;
                if (_RSTriAvailable > 0.5)
                {
                    half3 tri = SampleTriplanar(IN.positionWS, normal);
                    baseColor = tri.r >= 0 ? tri : IN.color.rgb;
                }
                else if (_RSNormalFallback > 0.5)
                {
                    baseColor = half3(normal * 0.5 + 0.5);
                }
                else
                {
                    baseColor = IN.color.rgb;
                }

                // 2. Apply freeze tint, then the preview cones (slot 0 drawn last: on top)
                baseColor = ApplyFreezeTint(baseColor, IN.positionWS);
                baseColor = ApplyPreviewCone(baseColor, IN.positionWS, _RSPreviewCone1A, _RSPreviewCone1B, _RSPreviewCone1C);
                baseColor = ApplyPreviewCone(baseColor, IN.positionWS, _RSPreviewCone0A, _RSPreviewCone0B, _RSPreviewCone0C);
                baseColor = ApplyBirthFade(baseColor, IN.color.a);

                // 2b. Leak tint: the free voxel in front of this surface point
                // touches void unknown — passthrough shows through here.
                if (_RSShowHoles > 0.5)
                    baseColor = lerp(baseColor, half3(1.0, 0.25, 0.1), saturate(IN.hole) * 0.85);

                // 3. Wireframe: discard interior, white edges blending to vertex color at vertices
                if (_RSWireframe > 0.5)
                {
                    float thickness = max(_RSWireThickness, 0.2);
                    float3 bary = IN.barycentric;
                    float3 dx = ddx(bary);
                    float3 dy = ddy(bary);
                    float3 edgeWidth = sqrt(dx * dx + dy * dy);
                    float3 edge = smoothstep(0.0, edgeWidth * thickness, bary);
                    float minEdge = min(edge.x, min(edge.y, edge.z));

                    // Discard interior — threshold scales inversely with thickness
                    float discardThreshold = saturate(1.0 - thickness * 0.15);
                    if (minEdge > discardThreshold)
                        discard;

                    // Vertex proximity: 1 at vertex, ~0.5 at edge midpoint
                    float vertexProximity = max(bary.x, max(bary.y, bary.z));
                    float vertBlend = smoothstep(0.35, 0.85, vertexProximity);
                    half3 wireColor = lerp(half3(0.9, 0.9, 0.92), baseColor, vertBlend);
                    return half4(wireColor, 1);
                }

                return half4(baseColor, 1);
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
            ZWrite On
            ColorMask 0

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "ScanMeshBirth.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 prevPos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
                uint   _pad;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                uint idx = _SurfaceIndices[vertID];
                GPUVertex gv = _SurfaceVerts[idx];
                float fade = ScanMeshBirthFade((gv.packedColor >> 24) / 255.0, gv.voxelFlatIdx);
                float3 pos = ScanMeshPresentedPos(gv.pos, gv.prevPos, gv.norm, fade);
                OUT.positionHCS = TransformWorldToHClip(pos);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode"="DepthNormals" }
            ZWrite On

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "ScanMeshBirth.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 prevPos;
                float3 norm;
                uint   packedColor;
                uint   voxelFlatIdx;
                uint   _pad;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint>      _SurfaceIndices;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 normalWS : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings OUT = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                uint idx = _SurfaceIndices[vertID];
                GPUVertex gv = _SurfaceVerts[idx];
                float fade = ScanMeshBirthFade((gv.packedColor >> 24) / 255.0, gv.voxelFlatIdx);
                float3 pos = ScanMeshPresentedPos(gv.pos, gv.prevPos, gv.norm, fade);
                OUT.positionHCS = TransformWorldToHClip(pos);
                OUT.normalWS    = gv.norm;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);
                float3 n = normalize(IN.normalWS);
                return half4(n * 0.5 + 0.5, 1);
            }
            ENDHLSL
        }
    }
}
