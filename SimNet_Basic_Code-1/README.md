# SimNet Basic Code

This repository contains sample code for training a simple SimNet model.

Files
- `simnet.py` - main model definition
- `train_simnet_on_toy_example1.py` - training script for a toy example

How to push from PyCharm
1. Open the project in PyCharm.
2. VCS -> Enable Version Control Integration -> Git.
3. Commit files (VCS -> Commit) and include `README.md` and `.gitignore`.
4. Add remote: VCS -> Git -> Remotes -> + -> paste remote URL (from GitHub/GitLab).
5. Push: VCS -> Git -> Push.

Alternatively, use terminal commands in the project's root:

```bash
cd /Users/yanivmazor3/PycharmProjects/Metasurfaces_OTA/SimNet_Basic_Code-1
git init
git add .
git commit -m "Initial commit"
git remote add origin <YOUR_REMOTE_URL>
git branch -M main
git push -u origin main
```

