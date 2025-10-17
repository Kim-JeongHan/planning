#!/bin/bash
# Generate all documentation images from examples

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Generating Documentation Images${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Set PYTHONPATH
export PYTHONPATH=/home/jeonghan/workspace/planning

# Change to project directory
cd /home/jeonghan/workspace/planning

# List of examples to run
examples=(
    "rrt_example"
    "rrt_connect_example"
    "rrt_star_example"
    "rrg_example"
    "prm_example"
    "prm_star_example"
)

total=${#examples[@]}
current=0

for example in "${examples[@]}"; do
    current=$((current + 1))

    echo -e "${YELLOW}[$current/$total]${NC} Running ${GREEN}$example${NC}..."

    # Run example with --save-image flag
    # Use timeout to automatically kill after 10 seconds
    timeout 10s uv run python examples/${example}.py --save-image 2>&1 | head -n 30 &

    # Get the PID of the background process
    pid=$!

    # Wait for 3 seconds
    sleep 3

    # Kill the process gracefully
    kill -SIGINT $pid 2>/dev/null || true

    # Wait for process to finish
    wait $pid 2>/dev/null || true

    echo -e "${GREEN}✓${NC} ${example} completed\n"

    # Small delay between examples
    sleep 0.5
done

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All images generated successfully!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo "Generated images in docs/images/:"
ls -lh docs/images/*.png 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'

echo -e "\n${YELLOW}Note:${NC} Images are saved with transparent background"
