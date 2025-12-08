#!/bin/bash
# Script pour fetch et pull toutes les branches d'un remote

REMOTE=${1:-origin}   # par défaut "origin", mais tu peux passer un autre remote en argument

echo "🔄 Fetch depuis $REMOTE..."
git fetch --all

# Boucle sur toutes les branches distantes
for branch in $(git branch -r | grep "$REMOTE/" | grep -v '\->'); do
    local_branch=${branch#"$REMOTE/"}

    echo "➡️ Mise à jour de la branche $local_branch"

    # Crée la branche locale si elle n'existe pas encore
    if ! git show-ref --verify --quiet refs/heads/$local_branch; then
        git checkout --track $branch
    else
        git checkout $local_branch
    fi

    # Pull les dernières modifications
    git pull $REMOTE $local_branch
done

echo "✅ Toutes les branches ont été mises à jour."
