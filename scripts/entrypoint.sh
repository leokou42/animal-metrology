#!/bin/bash
set -e

# Copy Depth Pro weights from build cache to mounted volume (if available and not already present)
if [ -f /opt/depth_pro/depth_pro.pt ] && [ ! -f /app/weights/depth_pro.pt ]; then
    echo "Copying depth_pro.pt from build cache to /app/weights/ ..."
    cp /opt/depth_pro/depth_pro.pt /app/weights/depth_pro.pt
    echo "Done."
fi

exec "$@"
