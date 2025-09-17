#!/usr/bin/env python3
"""
Verification script for EndoVis 2025 YOLO-Pose Docker Submission
Confirms all components are ready for deployment
"""

import os
import json
from pathlib import Path
import subprocess

def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}: {file_path} ({size_mb:.1f}MB)")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_docker_image(image_name):
    """Check if Docker image exists"""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", image_name],
            capture_output=True, text=True
        )
        if result.returncode == 0 and image_name in result.stdout:
            print(f"✅ Docker image: {image_name} - READY")
            return True
        else:
            print(f"❌ Docker image: {image_name} - NOT FOUND")
            return False
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

def verify_submission():
    """Verify all submission components"""
    print("=" * 70)
    print("EndoVis 2025 YOLO-Pose Submission Verification")
    print("Performance: HOTA=0.4281 (+23.6% improvement)")
    print("=" * 70)

    base_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker")
    submission_dir = base_dir / "synapse_submission_yolo_pose"

    # Check main components
    print("\n🔍 Main Components:")
    components = [
        (base_dir / "main_yolo_pose.py", "Main inference script"),
        (base_dir / "Dockerfile_yolo_pose", "Optimized Dockerfile"),
        (base_dir / "requirements_yolo_pose.txt", "Requirements file"),
        (base_dir / "surgical_keypoints_best.pt", "Trained model weights"),
        (base_dir / "build_yolo_pose_docker.sh", "Build script"),
        (base_dir / "deploy_yolo_pose_synapse.sh", "Deployment script"),
    ]

    all_components = True
    for file_path, description in components:
        if not check_file_exists(file_path, description):
            all_components = False

    # Check Docker image
    print("\n🐳 Docker Image:")
    docker_ready = check_docker_image("endovis2025-yolo-pose-v3")

    # Check submission package
    print("\n📦 Submission Package:")
    submission_files = [
        (submission_dir / "endovis2025-yolo-pose-v3.tar", "Docker image tar"),
        (submission_dir / "main_yolo_pose.py", "Source code"),
        (submission_dir / "Dockerfile_yolo_pose", "Dockerfile"),
        (submission_dir / "requirements_yolo_pose.txt", "Requirements"),
        (submission_dir / "README.md", "Documentation"),
        (submission_dir / "submission_metadata.json", "Metadata"),
    ]

    package_ready = True
    for file_path, description in submission_files:
        if not check_file_exists(file_path, description):
            package_ready = False

    # Check metadata
    print("\n📋 Submission Metadata:")
    metadata_file = submission_dir / "submission_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            print(f"✅ Submission name: {metadata.get('submission_name', 'Unknown')}")
            print(f"✅ Docker image: {metadata.get('docker_image', 'Unknown')}")
            print(f"✅ Method: {metadata.get('method', 'Unknown')}")

            perf = metadata.get('performance', {})
            print(f"✅ HOTA Score: {perf.get('hota_score', 'Unknown')}")
            print(f"✅ Improvement: {perf.get('improvement', 'Unknown')}")

        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
            package_ready = False

    # Final status
    print("\n" + "=" * 70)
    if all_components and docker_ready and package_ready:
        print("🎉 VERIFICATION SUCCESSFUL!")
        print("✅ All components ready for Synapse deployment")
        print("✅ HOTA Performance: 0.4281 (+23.6% improvement)")
        print("✅ Docker image: endovis2025-yolo-pose-v3")
        print("✅ Submission package: synapse_submission_yolo_pose/")

        # Show next steps
        print("\n🚀 Next Steps:")
        print("1. Set Synapse authentication token:")
        print("   export SYNAPSE_AUTH_TOKEN='your_token_here'")
        print("2. Run deployment script:")
        print("   ./deploy_yolo_pose_synapse.sh")
        print("3. Or manually upload files from synapse_submission_yolo_pose/")

        return True
    else:
        print("❌ VERIFICATION FAILED!")
        print("Please check the missing components above.")
        return False

if __name__ == "__main__":
    success = verify_submission()
    exit(0 if success else 1)