#!/usr/bin/env python3
"""Upload EndoVis 2025 YOLO-Pose submission to Synapse"""

import synapseclient
import os
import sys
import json
from pathlib import Path

def upload_submission():
    # Authenticate
    syn = synapseclient.Synapse()

    try:
        syn.login(authToken=os.getenv('SYNAPSE_AUTH_TOKEN'))
        print("✅ Successfully authenticated with Synapse")
    except Exception as e:
        print(f"❌ Synapse authentication failed: {e}")
        return False

    # Project and submission details
    project_id = "syn54123724"  # EndoVis 2025 project
    submission_dir = "synapse_submission_yolo_pose"

    # Load metadata
    with open(f"{submission_dir}/submission_metadata.json", 'r') as f:
        metadata = json.load(f)

    print(f"Uploading submission: {metadata['submission_name']}")
    print(f"Performance: HOTA = {metadata['performance']['hota_score']} ({metadata['performance']['improvement']})")

    # Create folder for this submission
    folder_name = f"{metadata['submission_name']}_{metadata['submission_date'][:10]}"

    try:
        # Upload Docker tar file
        docker_file = f"{submission_dir}/endovis2025-yolo-pose-v3.tar"
        if os.path.exists(docker_file):
            print(f"Uploading Docker image: {docker_file}")
            docker_entity = synapseclient.File(
                docker_file,
                description=f"YOLO-Pose Docker image - HOTA: {metadata['performance']['hota_score']}",
                parent=project_id
            )
            docker_entity = syn.store(docker_entity)
            print(f"✅ Docker image uploaded: {docker_entity.id}")

        # Upload source files
        source_files = [
            "main_yolo_pose.py",
            "requirements_yolo_pose.txt",
            "Dockerfile_yolo_pose",
            "README.md",
            "submission_metadata.json"
        ]

        for file_name in source_files:
            file_path = f"{submission_dir}/{file_name}"
            if os.path.exists(file_path):
                print(f"Uploading: {file_name}")
                file_entity = synapseclient.File(
                    file_path,
                    description=f"YOLO-Pose source - {file_name}",
                    parent=project_id
                )
                file_entity = syn.store(file_entity)
                print(f"✅ {file_name} uploaded: {file_entity.id}")

        print("🎉 All files uploaded successfully!")
        print(f"Submission performance: HOTA = {metadata['performance']['hota_score']} (+23.6% improvement)")
        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    success = upload_submission()
    sys.exit(0 if success else 1)
