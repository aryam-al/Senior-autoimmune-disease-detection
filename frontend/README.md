# ImmunoAI Frontend

Next.js app (original ImmunoAI design) connected to the ImmunoAI backend.

## Run

1. Start the backend first - it must be on http://localhost:8000.
2. Double-click `start-frontend.bat`, or run `npm install` then `npm run dev`.
3. Open http://localhost:3000

To use a different backend address, copy `.env.local.example` to `.env.local` and change `NEXT_PUBLIC_API_URL`.

## Pages

| Path | What it does |
|---|---|
| `/login`, `/register` | Sign in / sign up (email + password, and Google when the backend has `GOOGLE_CLIENT_ID`) |
| `/` | Home: system status, quick actions, recent reports |
| `/upload` | Upload a PDF; the values found are passed to the review screen |
| `/review` | Check / complete the 23 values (from a PDF, or `?mode=manual` to type them) and run the analysis |
| `/analysis/[id]` | XGBoost and Qwen results side by side, other possibilities, SHAP insights, lab values, download (print to PDF) |
| `/analysis` | Opens the latest analysis (Reports tab) |
| `/history` | All reports, delete |
| `/profile` | User, model accuracy and status, sign out |

## Code

- `lib/api.ts` - every backend call and type
- `lib/auth.tsx` - sign-in state and page protection
- `lib/markers.ts` - marker groups, units, normal ranges
- `components/` - original design components (header, bottom nav, confidence circle, impact bar) plus the new form and Google button
