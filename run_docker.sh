#!/usr/bin/env bash
# ==============================================================================
# run_docker.sh: Skrip Pembantu untuk Manajemen & Eksekusi Docker D3QN MEC
# ==============================================================================

set -e

IMAGE_NAME="d3qn-mec:latest"
WORKSPACE_DIR="$(pwd)"

print_help() {
    echo "=========================================================================="
    echo "           D3QN Mobile Edge Computing - Docker Execution Helper           "
    echo "=========================================================================="
    echo "Penggunaan: ./run_docker.sh [COMMAND] [ARGS...]"
    echo ""
    echo "Perintah yang Tersedia:"
    echo "  build                     Build image Docker ($IMAGE_NAME)"
    echo "  run [config_file]         Jalankan pipeline eksperimen dengan file .cfg"
    echo "                            (Default: paper_aligned_code/configs/sudden_shift_reset.cfg)"
    echo "  train [config_file]       Jalankan training paper-aligned saja"
    echo "  legacy                    Jalankan kode legacy (TensorFlow 1.15.5)"
    echo "  shell                     Buka interactive bash shell di dalam container"
    echo "  help                      Tampilkan bantuan ini"
    echo ""
    echo "Contoh:"
    echo "  ./run_docker.sh build"
    echo "  ./run_docker.sh run paper_aligned_code/configs/stationary.cfg"
    echo "  ./run_docker.sh run paper_aligned_code/configs/sudden_shift.cfg"
    echo "  ./run_docker.sh shell"
    echo "=========================================================================="
}

cmd="${1:-help}"

case "$cmd" in
    build)
        echo "[DOCKER] Membangun image Docker '$IMAGE_NAME'..."
        docker build -t "$IMAGE_NAME" .
        echo "[DOCKER] Selesai! Image '$IMAGE_NAME' siap digunakan."
        ;;

    run)
        config_path="${2:-paper_aligned_code/configs/sudden_shift_reset.cfg}"
        shift 2 || shift 1
        echo "[DOCKER] Menjalankan eksperimen dengan config: $config_path"
        docker run --rm -it \
            -v "$WORKSPACE_DIR":/app \
            -w /app \
            "$IMAGE_NAME" \
            python3 paper_aligned_code/run_experiment.py --config "$config_path" "$@"
        ;;

    train)
        config_path="${2:-paper_aligned_code/configs/sudden_shift.cfg}"
        shift 2 || shift 1
        echo "[DOCKER] Menjalankan training dengan config: $config_path"
        docker run --rm -it \
            -v "$WORKSPACE_DIR":/app \
            -w /app \
            "$IMAGE_NAME" \
            python3 paper_aligned_code/train.py --config "$config_path" "$@"
        ;;

    legacy)
        echo "[DOCKER] Menjalankan kode legacy (TensorFlow 1.15.5)..."
        docker run --rm -it \
            -v "$WORKSPACE_DIR":/app \
            -w /app \
            "$IMAGE_NAME" \
            python train.py
        ;;

    shell|bash)
        echo "[DOCKER] Membuka interactive bash shell..."
        docker run --rm -it \
            -v "$WORKSPACE_DIR":/app \
            -w /app \
            "$IMAGE_NAME" \
            /bin/bash
        ;;

    help|--help|-h)
        print_help
        ;;

    *)
        echo "[ERROR] Perintah tidak dikenali: $cmd"
        print_help
        exit 1
        ;;
esac

