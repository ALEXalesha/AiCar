#!/usr/bin/env bash
# Публикация на GitHub: https://github.com/ALEXalesha/AiCar
#
#   bash tools/publish_github.sh            # только код
#   bash tools/publish_github.sh v1.0.0     # код и тег
#   DRY_RUN=1 bash tools/publish_github.sh  # собрать и проверить ветку, ничего не пушить
#
# Источник правды - Gitea. На GitHub уезжает копия main, в которой личный gmail автора и
# коммитера заменён на анонимную почту аккаунта GitHub: почта автора видна любому, кто
# откроет коммит. Рабочая история не трогается, ветка пересобирается каждый раз заново -
# отсюда --force.
#
# Теги. Выпуски на GitHub привязаны к тегам. Когда чистка задевает старые коммиты, теги
# на GitHub остаются на коммитах ДО неё, и личное видно через них. Поэтому каждый тег на
# GitHub, который не ведёт в свежую ветку, переставляется на её коммит с тем же временем
# и заголовком. Выпуск при этом остаётся на месте со всеми файлами.
set -euo pipefail

export PATH="$PATH:/c/Program Files/GitHub CLI"
REPO=ALEXalesha/AiCar
# Личный адрес не записан здесь строкой: скрипт берёт его из настроек репозитория.
# Иначе выходило смешно - файл, который убирает почту автора из коммитов, публиковал
# её открытым текстом на первой же странице.
PRIVATE_EMAIL="$(git config user.email)"
PUBLIC_EMAIL=203467574+ALEXalesha@users.noreply.github.com
# Второе, чего не должно быть в открытом репозитории: адрес домашнего Gitea. Он лежит
# в плане и спецификации из docs/superpowers, в строчке про remote origin. И он тоже
# берётся из настроек, а не пишется строкой: в первой версии скрипт нашёл адрес в
# самом себе и переписал собственную замену в бессмысленную gitea.local -> gitea.local.
LAN_GITEA="$(git remote get-url origin | sed -E 's#^[a-z]+://([^/]+)/.*#\1#')"
PUBLIC_GITEA='gitea.local'
TAG="${1:-}"

cd "$(dirname "$0")/.."

# Третье - личный путь: папка автора на диске, в ней его имя (стоял в docs/modules/build.md).
# Тоже из настроек: git config publish.privatePath, вида C:\папка\имя. Из него строится
# выражение, которое ловит любое написание - прямую, обратную и удвоенную обратную косую
# черту, любой регистр; заменяется на нейтральное Users\you. Выражение и замена лежат
# файлами в .git: в строку filter-branch обратные косые черты пришлось бы экранировать
# в три слоя.
esc() { printf '%s' "$1" | sed 's/[.[\*^$+?(){}|]/\\&/g'; }
PRIVATE_PATH="$(git config --get publish.privatePath || true)"
if [ -z "$PRIVATE_PATH" ]; then
  echo "не задано git config publish.privatePath - без него личный путь не заменить" >&2
  exit 1
fi
IFS='\/' read -r -a PATH_PARTS <<< "${PRIVATE_PATH#?:}"
PATH_RE=""
n=0
for part in "${PATH_PARTS[@]}"; do
  [ -n "$part" ] || continue
  case $n in
    0) PATH_RE="$(esc "$part")" ;;
    1) PATH_RE="$PATH_RE"'([\\/]+)'"$(esc "$part")" ;;
    *) PATH_RE="$PATH_RE"'[\\/]+'"$(esc "$part")" ;;
  esac
  n=$((n + 1))
done
if [ $n -lt 2 ]; then
  echo "publish.privatePath: нужны хотя бы две папки, вида C:\\папка\\имя" >&2
  exit 1
fi
GIT_DIR_ABS="$(cd "$(git rev-parse --git-dir)" && pwd)"
PATH_SED="$GIT_DIR_ABS/publish-path.sed"
PATH_GREP="$GIT_DIR_ABS/publish-path.grep"
printf '%s\n' "s#$PATH_RE#Users\\1you#gI" > "$PATH_SED"
printf '%s\n' "$PATH_RE" > "$PATH_GREP"

git remote get-url github >/dev/null 2>&1 || git remote add github "https://github.com/$REPO.git"

git branch -f github-main main
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter "
  if [ \"\$GIT_AUTHOR_EMAIL\" = '$PRIVATE_EMAIL' ]; then export GIT_AUTHOR_EMAIL='$PUBLIC_EMAIL'; fi
  if [ \"\$GIT_COMMITTER_EMAIL\" = '$PRIVATE_EMAIL' ]; then export GIT_COMMITTER_EMAIL='$PUBLIC_EMAIL'; fi
" --msg-filter "sed -E -f '$PATH_SED'" github-main >/dev/null 2>&1
git update-ref -d refs/original/refs/heads/github-main 2>/dev/null || true

# Личное убирается и из содержимого файлов по всей публикуемой истории.
#
# Замена почты автора прячет её в метаданных коммита, а строка в файле остаётся на виду -
# именно так адрес однажды и уехал на GitHub внутри самого скрипта публикации. Правится
# только то, где он действительно встречается: блобы этих путей переписываются прямо в
# индексе, без выгрузки дерева на диск.
# `|| true`: без совпадений git grep возвращает 1, и при pipefail скрипт молча
# обрывался бы здесь.
DIRTY=$( { git grep -I -l -E "$PRIVATE_EMAIL|$LAN_GITEA" $(git rev-list github-main) -- 2>/dev/null || true
           git grep -I -l -i -E -f "$PATH_GREP" $(git rev-list github-main) -- 2>/dev/null || true; } \
        | sed 's/^[^:]*://' | sort -u | tr '\n' ' ' || true)
if [ -n "$DIRTY" ]; then
  echo "личное в файлах: $DIRTY- переписываю содержимое"
  FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --index-filter "
    for f in $DIRTY; do
      mode=\$(git ls-tree \$GIT_COMMIT -- \"\$f\" | awk '{print \$1}')
      [ -n \"\$mode\" ] || continue
      blob=\$(git cat-file blob \$GIT_COMMIT:\"\$f\" \
             | sed -e 's/$PRIVATE_EMAIL/$PUBLIC_EMAIL/g' -e 's/$LAN_GITEA/$PUBLIC_GITEA/g' \
             | sed -E -f '$PATH_SED' \
             | git hash-object -w --stdin)
      git update-index --cacheinfo \$mode,\$blob,\"\$f\"
    done
  " github-main >/dev/null 2>&1
  git update-ref -d refs/original/refs/heads/github-main 2>/dev/null || true
fi

# grep без -q: с -q он выходит на первом совпадении, git log получает SIGPIPE, и при
# pipefail условие становится ложным - страж молча пропускал бы найденное.
if git log github-main --format='%ae%n%ce' | grep -x "$PRIVATE_EMAIL" >/dev/null; then
  echo "в публикуемой ветке остался личный адрес в авторе коммита - пуш отменён" >&2
  exit 1
fi

# И в самих файлах - по всей публикуемой истории, а не только в её вершине.
# Проверка появилась после того, как личный адрес уехал на GitHub открытым текстом
# внутри этого самого скрипта: почта автора в коммитах пряталась, а строка в файле
# лежала на виду.
if git grep -I -q -E "$PRIVATE_EMAIL|$LAN_GITEA" $(git rev-list github-main) -- 2>/dev/null; then
  echo "личное осталось в файлах публикуемой истории - пуш отменён" >&2
  git grep -I -l -E "$PRIVATE_EMAIL|$LAN_GITEA" $(git rev-list github-main) -- | head -5 >&2
  exit 1
fi
if git grep -I -q -i -E -f "$PATH_GREP" $(git rev-list github-main) -- 2>/dev/null; then
  echo "личный путь остался в файлах публикуемой истории - пуш отменён" >&2
  exit 1
fi
if git log github-main --format=%B | grep -i -E -f "$PATH_GREP" >/dev/null; then
  echo "личный путь остался в сообщениях коммитов - пуш отменён" >&2
  exit 1
fi

# Теги на GitHub, которые не ведут в свежую ветку, получают пару по времени и заголовку.
MOVES=""
git for-each-ref --format='%(refname)' refs/github-tags | xargs -r -n1 git update-ref -d
git fetch -q github '+refs/tags/*:refs/github-tags/*' 2>/dev/null || true
for ref in $(git for-each-ref --format='%(refname)' refs/github-tags); do
  name=${ref#refs/github-tags/}
  old=$(git rev-parse "$ref^{commit}")
  git merge-base --is-ancestor "$old" github-main && continue
  key=$(git log -1 --format='%at %s' "$old")
  # awk дочитывает вход до конца, без exit: при раннем выходе git log получал SIGPIPE,
  # и с pipefail скрипт обрывался здесь молча (код 141).
  new=$(git log github-main --format='%H %at %s' \
        | awk -v k="$key" '{h=$1; $1=""; if (!f && substr($0,2)==k) {print h; f=1}}')
  if [ -z "$new" ]; then
    echo "тег $name: в публикуемой ветке нет коммита с тем же временем и заголовком - оставлен" >&2
    continue
  fi
  echo "тег $name: $(git rev-parse --short "$old") -> $(git rev-parse --short "$new")"
  MOVES="$MOVES +$new:refs/tags/$name"
done
git for-each-ref --format='%(refname)' refs/github-tags | xargs -r -n1 git update-ref -d

echo "ветка github-main: $(git rev-list --count github-main) коммитов, $(git log -1 --format=%h github-main)"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "DRY_RUN=1: ветка собрана и проверена, на GitHub ничего не отправлено"
  [ -z "$MOVES" ] || echo "теги переставились бы:$MOVES"
  exit 0
fi
gh auth setup-git
git push github github-main:main --force
if [ -n "$MOVES" ]; then
  git push github $MOVES
fi

if [ -n "$TAG" ]; then
  # Тег ставится прямо на GitHub, без локального: локальный указывал бы на
  # переписанный коммит github-main и путался с тегами Gitea.
  git push github "+github-main:refs/tags/$TAG"
  echo "тег $TAG отправлен"
fi
