# Contributing to SteelSight

Thank you for your interest in contributing to **SteelSight** (Automated Industrial Surface Defect Inspection Platform)! We welcome contributions in computer vision, model optimization, web UI/UX, and edge deployment.

---

## 🛠️ Code of Conduct
Please be polite, collaborative, and constructive when raising issues or reviewing code.

---

## 🚀 How to Contribute

### 1. Reporting Bugs
- Search existing GitHub Issues before opening a new one.
- Provide a clear, reproducible example with operating system, Python/Node versions, and exact command or stack trace.

### 2. Feature Requests
- Propose new model architectures (e.g. YOLOv10, Vision Transformers / Swin-T, Diffusion-based anomaly detection).
- Suggest edge acceleration improvements (TensorRT, ONNX Runtime, OpenVINO, CoreML).
- File an issue outlining the motivation and architectural approach before starting large PRs.

### 3. Submitting Pull Requests
1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Verify Environment**:
   ```bash
   python test_environment.py
   ```
4. **Follow Formatting Guidelines**:
   - Python code formatted with `black` and PEP 8 conventions.
   - Frontend components formatted with Prettier/ESLint.
   - Type hints for all new public functions and API endpoints.
5. **Commit Messages**:
   Use conventional commits (`feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`).
6. **Push & Open PR**:
   Push to your fork and submit a Pull Request targeting `main`. Describe your changes and link any related issues.

---

## 🧪 Testing Guidelines
- If adding new computer vision transformations, verify backwards compatibility with `src/dataset.py`.
- If modifying FastAPI endpoints, test schemas using Swagger UI (`/docs`).
- If adding frontend features in `steelsight/`, run `npm run build` to verify type safety and build output.

Thank you for helping push industrial AI forward! 🏭
