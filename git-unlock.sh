#!/bin/bash
set -e

# 1. 检查是否在 Git 仓库内
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "错误：当前不在 Git 仓库目录中"
    exit 1
fi

# 2. 自动获取 Git 仓库根目录和当前分支
GIT_ROOT=$(git rev-parse --show-toplevel)
CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "HEAD")

# 3. 定义所有常见锁文件路径
LOCK_FILES=(
    "$GIT_ROOT/.git/index.lock"       # 暂存区索引锁（最常见）
    "$GIT_ROOT/.git/config.lock"      # 配置文件锁（本次推送遇到的错误）
    "$GIT_ROOT/.git/packed-refs.lock" # 打包引用锁
)

# 仅在正常分支（非分离头指针）下，加入对应分支的引用锁
if [ "$CURRENT_BRANCH" != "HEAD" ]; then
    LOCK_FILES+=("$GIT_ROOT/.git/refs/heads/${CURRENT_BRANCH}.lock")
fi

# 4. 遍历清理锁文件
echo "开始清理 Git 残留锁文件..."
echo "仓库路径: $GIT_ROOT"
echo "当前分支: $CURRENT_BRANCH"
echo "----------------------------------------"

for lock_file in "${LOCK_FILES[@]}"; do
    # 截取相对路径，输出更简洁
    relative_path="${lock_file#$GIT_ROOT/.git/}"
    if [ -f "$lock_file" ]; then
        rm -f "$lock_file"
        echo "  ✓ 已清理: $relative_path"
    else
        echo "  - 无残留: $relative_path"
    fi
done

echo "----------------------------------------"
echo "锁文件清理完成"
echo ""
echo "提示：若操作后仍报错，请先检查是否有正在运行的 Git 进程"
echo "      检查命令: ps aux | grep git"
