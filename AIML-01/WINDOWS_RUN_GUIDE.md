# 🪟 Run Fraud Platform on Windows — Step by Step

You are already in the right folder. Follow these steps in **Command Prompt (CMD)**.

---

## STEP 1 — Go inside the project folder

In your CMD window (where you see `E:\nexvia intership\ALML-01>`), type:

```cmd
cd fraud-platform
```

Press Enter. You should now see:

```
E:\nexvia intership\ALML-01\fraud-platform>
```

Check files are there:

```cmd
dir
```

You must see: `package.json`, `app`, `lib`, `backend`, `README.md`

---

## STEP 2 — Check Node.js is installed

```cmd
node -v
npm -v
```

- ✅ If you see versions like `v20.x.x` and `10.x.x` → go to Step 3.
- ❌ If you see `'node' is not recognized` → install Node.js:
  1. Go to https://nodejs.org → Download **LTS (v20)** Windows installer
  2. Install it (keep clicking Next)
  3. **Close CMD and open a NEW CMD**, then run `node -v` again

---

## STEP 3 — Install project files (first time only)

```cmd
npm install
```

⏳ Wait 1–2 minutes. It will download Next.js + React. Do NOT close the window.
You are done when you see `added XXX packages` and the prompt returns.

---

## STEP 4 — Start the app (development mode)

```cmd
npm run dev
```

Wait until you see:

```
▲ Next.js 14.2.5
- Local: http://localhost:3000
✓ Ready
```

**Leave this window open.** Do not press Ctrl+C.

---

## STEP 5 — Open in browser

Open Chrome/Edge and go to:

```
http://localhost:3000
```

You will see the Fraud Detection Platform dashboard with 6 tabs:
- **Overview** — KPIs + charts
- **Live Stream** — click ▶ Start stream
- **Alert Queue** — click an alert → Confirm fraud / False positive
- **Model Lab** — click Recompute live PR-AUC
- **Monitoring** — click Run drift check
- **Pipeline & Docs** — deploy instructions

🎉 **Done! The app is running.**

### To stop the app
Press `Ctrl + C` in the CMD window, then type `Y` and Enter.

### To run again next time
Just repeat Step 1 → Step 4 (skip Step 3 if you already did `npm install`).

---

## (Optional) Production mode — faster, like Vercel

```cmd
npm run build
npm start
```

Then open `http://localhost:3000` again.

---

## (Optional) Deploy to GitHub + Vercel

### A. Push to GitHub
1. Create a new empty repo on github.com → name it `fraud-platform` (do NOT add README)
2. In CMD, inside `fraud-platform` folder:

```cmd
git init
git add .
git commit -m "Fraud platform"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fraud-platform.git
git push -u origin main
```

(Replace `YOUR-USERNAME` with your GitHub username. If Git is not installed, download from https://git-scm.com)

### B. Deploy on Vercel (free)
1. Go to https://vercel.com → Sign up / Log in with GitHub
2. Click **Add New → Project**
3. Select your `fraud-platform` repo → **Import**
4. Framework: **Next.js** (auto-detected) → Click **Deploy**
5. Wait 1–2 minutes → you get a live link like `fraud-platform.vercel.app` 🎉

No settings or env vars needed.

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `npm` is not recognized | Install Node.js LTS (Step 2), reopen CMD |
| `npm install` fails / takes long | Check internet, run `npm install` again |
| Port 3000 already in use | Close other apps, or run `npm run dev -- -p 3001` and open `http://localhost:3001` |
| Page shows blank | Make sure `npm run dev` window is still open and says ✓ Ready |
| PowerShell errors | Use **CMD** (black window), not PowerShell |
| `git` is not recognized | Install Git from https://git-scm.com, reopen CMD |

---

## 🐍 Full Python backend (optional, advanced)

Only needed for the real ML training (Feast/XGBoost/Kafka). The web app above works **without** it.

```cmd
cd backend
docker compose up -d
python -m fraud.data.prepare
```

Full commands: see `backend/README.md`. Requires Docker Desktop + Kaggle data download.
