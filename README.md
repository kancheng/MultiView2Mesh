# MultiView2Mesh
A multi-stage vision pipeline integrating SAM, Mask2Former, Zero123, and TripoSR for multi-view segmentation, 3D reconstruction, and automatic material replacement.


A. Material Segmentation

A1: SAM Segmentation Demo
- One handbag image
- Use SAM
- Submit: original, masks, overlay
- Write 300–500 word analysis

A2: Multi-view Segmentation
- 3 different angles
- Mask2Former / SegFormer
- Submit masks + part standardization

A3: Multimodal Segmentation
- Generate 8 views with Zero123
- Segment all views + simple mask aggregation

B. 3D GenAI Training

B1: Zero123 Multi-view Generation
- 1 handbag image → 8–12 views
- Analyze model limitations

B2: Zero123 → TripoSR → Mesh
- Reconstruct mesh
- Submit mesh + screenshots
- Analyze holes, UV, normals

B3: Automatic Material Replacement
- Map masks to mesh UV
- Replace texture and render
