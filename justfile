# Print all available commands
default:
  just --list

# Sync configuration to the remote Obox
sync-obox:
  rsync -av --delete \
    --filter=':- .gitignore' \
    --exclude '.mypy_cache' --exclude '.ruff_cache' \
    --exclude '.jj' --exclude '.git' \
    . f44@192.168.1.144:/home/f44/zinc
