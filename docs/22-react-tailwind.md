# 22 — React & Tailwind: versions, architecture and the model to train

*Last verified: 25 October 2026. Version facts were checked against the npm registry, React and Tailwind release posts, the TypeScript 7 announcement, the Vite 8 announcement, the React Router v8 changelog and Node's release index.*

> If you want one Qwen3 base for Kali, Java and React instead of a React-specific coder, see [26 — One Qwen3 for all](26-one-model-for-all.md). The React-specific adapter here is still the higher-quality option for React.

This is the start of the front-end track. It answers the same three questions as the Java track:

1. **Which React and Tailwind versions should the model target?**
2. **What does a good React/Tailwind codebase look like, concretely?**
3. **Which Qwen coder can actually be fine-tuned on a 32 GB M1 Pro?**

The short answers:

- **React 19.3 + Tailwind CSS 4.3 + Vite 8 + Node 24 LTS**, with a TypeScript 7-compatible `tsconfig` but a TypeScript 6 tooling bridge for ESLint.
- **Feature-based architecture, a token-driven design system, accessible primitives, server-state discipline and tested states** — not just “utility classes in JSX”.
- **Qwen2.5-Coder-14B-Instruct 4-bit QLoRA** locally; **Qwen3-Coder-30B-A3B** as teacher/judge. A text-only coder cannot see screenshots; use a VLM to turn a reference into a written design spec, then train the coder on the spec.

---

## 1. Which “latest” do you target?

The JavaScript ecosystem ships fast, and four different version axes matter: React, Tailwind, the build tool, and the type checker.

| Component | Latest as of 25 Oct 2026 | Status | Use in this handbook |
|---|---|---|---|
| **React / React DOM** | **19.3.1** (14 Oct 2026) | Stable; 19.3.0 shipped 9 Sep 2026 | **Primary target** |
| **Tailwind CSS** | **4.3.3** (16 Jul 2026) | Stable; 4.3.0 shipped 8 May 2026 | **Primary target** |
| **Vite** | **8.3.5** (21 Oct 2026) | Stable; 8.0 shipped 12 Mar 2026 | **Primary target** |
| **Node.js** | **24.22.1 “Krypton”** (20 Oct 2026) | **Active LTS** | **Primary runtime** |
| Node.js | 26.12.1 (21 Oct 2026) | Current, not LTS yet in this window | Do not require for a training target |
| **TypeScript** | **7.0.2** (8 Jul 2026) | Stable native compiler; 7.1 Beta 20 Oct 2026 | **Source/type-check target** |
| **React Router** | **8.5.0** (19 Oct 2026) | Stable; 8.0 shipped 17 Jun 2026 | **Routing target** |
| **TanStack Query** | 5.104.5 | Stable | Server-state target |
| **Storybook** | 10.6.3 (20 Oct 2026) | Stable; ESM-only since 10.0 | Optional component workshop |
| **Vitest** | 5.0.5 (15 Oct 2026) | Stable | Unit/component tests |
| **Playwright** | 1.64.1 | Stable | E2E, screenshots, axe |
| **ESLint** | 10.13.0 (16 Oct 2026) | Stable | Lint target |
| **Motion** | 14.6.0 | Stable | Optional animation |
| **shadcn/ui** | current | MIT component patterns | Design-system reference |

### 1.1 React 19.3

React 19.3 is a minor release with real features:

- **View Transitions** are stable (`<ViewTransition>`), so page and list transitions can use the browser's View Transition API.
- **Fragment refs** are stable.
- **`browser()`** in React DOM helps with client-only rendering and bailouts.
- **Trusted Types** support lets React pass Trusted Types objects through to DOM APIs.
- Server Components remain stable, but their bundler/framework APIs are not semver-stable between 19.x minors. This handbook does **not** train a Server Components stack.

React 19 already gave us Actions, `useActionState`, `useFormStatus`, `useOptimistic`, `use`, `ref` as a prop, ref cleanup functions, and `<Context>` as a provider. Those are the patterns the model should learn.

### 1.2 Tailwind CSS 4.3

Tailwind 4 is **CSS-first**. The old `tailwind.config.js` + `@tailwind base/components/utilities` setup is not the default any more:

```css
@import "tailwindcss";

@theme {
  --color-brand: oklch(0.58 0.19 275);
  --radius-card: 1rem;
}
```

Tailwind 4.3 adds:

- first-party **scrollbar** utilities (`scrollbar-*`, `scrollbar-color-*`, `scrollbar-gutter-*`);
- **`@container-size`** for block-size container queries (`cqb`/`cqh`);
- **`zoom-*`** and **`tab-*`** utilities;
- stacked and compound **`@variant`** rules;
- new colors and logical-property utilities.

The v4 migration also renamed utilities and changed defaults. The ones that matter for training data:

| v3 | v4 |
|---|---|
| `shadow-sm` | `shadow-xs` |
| `shadow` | `shadow-sm` |
| `rounded-sm` | `rounded-xs` |
| `rounded` | `rounded-sm` |
| `blur-sm` | `blur-xs` |
| `outline-none` | `outline-hidden` (old behaviour); `outline-none` now really removes the outline |
| `ring` (3px) | `ring-3`; v4's `ring` is 1px |
| `bg-opacity-50` | `bg-black/50` |
| `flex-shrink-*` | `shrink-*` |
| `flex-grow-*` | `grow-*` |
| `overflow-ellipsis` | `text-ellipsis` |
| `decoration-slice` / `clone` | `box-decoration-slice` / `clone` |

Also: the default border color changed from `gray-200` to `currentColor`; `@apply` in CSS modules needs `@reference`; and the v4 browser baseline is Safari 16.4+, Chrome 111+, Firefox 128+.

**Tag every example with `tailwind: "3"` or `tailwind: "4"`.** A model trained on a mix of `@tailwind base` and `@import "tailwindcss"` without tags will produce a plausible but broken hybrid.

### 1.3 TypeScript 7 — the tooling trap

TypeScript 7.0 is the Go-native compiler: 8–12× faster type-checking, lower memory, stricter defaults.

It is **not a drop-in upgrade**:

- `strict` is on by default; `module` defaults to `esnext`.
- `target: es5`, `downlevelIteration`, `moduleResolution: node/node10`, `module: amd/umd/systemjs/none`, and `baseUrl` are removed.
- `types` and `rootDir` defaults are stricter; list your `@types` explicitly.
- The programmatic compiler API is **not available until 7.1**. That breaks tools that import TypeScript, including `typescript-eslint`, some webpack loaders and `ts-morph`.

For this handbook:

- **Write source and `tsconfig` for TypeScript 7.0.2** (no `baseUrl`, strict, explicit `types`).
- **Run ESLint with the TypeScript 6 API** (`@typescript/typescript6` alias or a pinned `typescript@6.0.3` dev dependency) until the lint stack supports 7.1.
- Do **not** train a model to use removed `tsconfig` options; those examples will not build on a 2026 project.

### 1.4 Vite 8 and React Router 8

Vite 8 replaces esbuild + Rollup with **Rolldown + Oxc**. Most plugins work, builds are much faster, and the default browser target moved to Chrome 111 / Edge 111 / Firefox 114 / Safari 16.4. A few Rollup-only hooks and `system`/`amd` output formats are gone.

React Router 8 is ESM-only, requires Node 22+, Vite 7+, React 19+, and **removed `react-router-dom`**. Import core APIs from `react-router` and browser-specific utilities from `react-router/dom`. It has data, declarative and framework modes; this handbook targets **data mode** in a Vite SPA.

Do not mix React Router 7 or Remix idioms into v8 examples without a migration tag.

### 1.5 The baseline config

A TypeScript 7-compatible app `tsconfig.json` (no `baseUrl`, strict, explicit `types`):

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023", "DOM", "DOM.Iterable"],
    "module": "preserve",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    "skipLibCheck": true,
    "noEmit": true,
    "types": ["vite/client"],
    "rootDir": "src",
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src"]
}
```

`vite.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
});
```

`package.json` scripts:

```json
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint .",
    "test": "vitest",
    "build": "vite build",
    "preview": "vite preview",
    "e2e": "playwright test"
  }
}
```

For ESLint, keep a TypeScript 6 API available. The practical bridge is a separate lint workspace pinned to `typescript@6.0.3`, or the alias package:

```json
{
  "devDependencies": {
    "typescript": "7.0.2",
    "@typescript/typescript6": "npm:typescript@6.0.2"
  }
}
```

`tsc` from TypeScript 7 checks the app; the lint workspace/alias supplies the 6.0 API. Do not put `typescript@7` in a plugin that imports the compiler API until 7.1 ships it.

### 1.6 Why Node 24 LTS, not Node 26

Node 24 is the active LTS in this window. Node 26 is Current. For a reproducible training and verification environment, target **Node 24 LTS**; it satisfies Vite 8, React Router 8 and Storybook 10. Mention Node 26 as an optional future upgrade.

---

## 2. Why a Vite SPA, not Next.js

“React and Tailwind” can mean many stacks. This handbook deliberately fixes one:

```text
React 19.3 + TypeScript 7 + Vite 8 + Tailwind 4.3
React Router 8 (data mode) + TanStack Query 5
Radix primitives + shadcn/ui patterns + CVA + tailwind-merge
Vitest 5 + Testing Library + Playwright 1.64 + axe-core
```

Why:

- The target is **front-end UI engineering**, not full-stack React.
- A Vite SPA has one rendering model: client components. Server Components, server actions, route handlers and `next/*` imports would add a second, conflicting vocabulary.
- Vite is the common base for React Router, Storybook, Vitest and Playwright; verification is simpler.
- A 14B model can learn one coherent set of idioms. Mixing Next.js App Router, React Router framework mode and Vite SPA data fetching produces confused code.

If you need Next.js later, make it a **separate adapter and dataset**, not more rows in this one.

---

## 3. Which Qwen coder?

### 3.1 Local training target

| Model | Params | Context | Local training on 32 GB | Role |
|---|---:|---:|---|---|
| **Qwen2.5-Coder-14B-Instruct** | 14.7B | 32K configured, 128K with RoPE scaling | **Yes**, 4-bit QLoRA, 24 of 48 blocks | **Primary local adapter** |
| Qwen2.5-Coder-7B-Instruct | 7.6B | 32K / 128K | Yes, all layers, seq 4096 | Smoke test, long-context iteration |
| Qwen2.5-Coder-32B-Instruct | 32.5B | 32K / 128K | Inference only at 4-bit | Occasional judge |
| Qwen3-Coder-30B-A3B-Instruct | 30.5B total / 3.3B active | 256K native | **No** — inference/teacher only | Teacher, judge, bulk generation |
| Qwen3-Coder-Next | 80B total / 3B active | 256K | No | API/rented GPU |
| Qwen3-14B | 14B | 32K | Yes | General fallback, less code-specialised |

**Qwen2.5-Coder-14B** is the best balance of code quality, React/TS knowledge, Apache-2.0 licensing and local trainability. It has standard attention, no Gated DeltaNet training problem, and a 4-bit MLX conversion.

**Qwen3-Coder-30B-A3B** is a better coder and agent than the 14B, but 30.5B total at 4-bit is ~18–20 GB before activations. Training a 30B MoE with LoRA on 32 GB is not a reliable MLX path. Use it to generate and grade React/Tailwind data, or full fine-tune it on a rented 80 GB GPU.

### 3.2 The vision limitation

Qwen2.5-Coder is **text-only**. It cannot consume a screenshot.

That matters because “make it look like this” is a visual task. Two honest options:

1. **Text-spec pipeline (this handbook).** A VLM or a human turns the screenshot into a structured `design.md`: layout, hierarchy, palette, type scale, spacing, components, states, breakpoints, copy. The coder is trained on the spec → code pair. At inference, the user supplies a brief or a VLM-generated spec.
2. **Separate VLM adapter.** If you need image → code end-to-end, fine-tune a vision-language model (Qwen3-VL class, or a Design2Code-style model). That is a different dataset, architecture and training run; do not pretend a text coder can see.

The synthetic chapter shows how to use a VLM as a **captioner/spec writer**, not as the student.

### 3.3 Memory sketch

| Setup | Weights | Adapter + optimizer | Peak estimate |
|---|---:|---:|---:|
| Qwen2.5-Coder-7B 4-bit, all 28 blocks, seq 4096 | ~4.5 GB | ~0.3 GB | 12–16 GB |
| Qwen2.5-Coder-14B 4-bit, 24/48 blocks, seq 2048 | ~8.7 GB | ~0.4 GB | 18–23 GB |
| Qwen2.5-Coder-14B 4-bit, 24/48 blocks, seq 3072 | ~8.7 GB | ~0.4 GB | 21–25 GB |
| Qwen2.5-Coder-14B 4-bit, all 48 blocks, seq 2048 | ~8.7 GB | ~0.8 GB | 21–26 GB |
| Qwen2.5-Coder-14B 4-bit, 16/48 blocks, seq 4096 | ~8.7 GB | ~0.3 GB | 21–25 GB |

These are estimates. Measure with `mlx_lm.lora` and `mx.get_peak_memory()`. Front-end tasks often need more context than a single Java class because they include component code, styles, tests and a design spec; that is why the 7B long-context run exists.

---

## 4. What the model must learn

| Layer | Skills | Examples |
|---|---|---|
| **React** | function components, hooks, `ref` as prop, Actions, `useActionState`, `useOptimistic`, Suspense, `use`, View Transitions | forms, dialogs, lists, optimistic updates |
| **TypeScript** | discriminated unions, generics, `satisfies`, narrowing, exhaustive checks | typed props, API schemas, reducers |
| **Tailwind 4** | CSS-first theme, tokens, variants, states, responsive, container queries, dark mode | buttons, cards, tables, layouts |
| **Architecture** | feature folders, primitives/patterns/pages, composition, server vs client state | app shells, design systems, refactors |
| **Design** | hierarchy, spacing, type scale, palette, density, motion, empty/loading/error states | dashboards, marketing pages, settings |
| **Accessibility** | semantic HTML, labels, focus, keyboard, contrast, target size, reduced motion | forms, menus, dialogs, tables |
| **Engineering** | Vite, Vitest, Testing Library, Playwright, axe, lint, typecheck, build | tests, bug fixes, migrations |
| **Agent behaviour** | read before edit, patch format, run checks, screenshot, recover, stop | tool trajectories |

The model does not need to memorize the whole Tailwind class list. It needs to know the **shape of a good component**, the **token system**, and **how to verify**.

---

## 5. Architecture for a React/Tailwind codebase

### 5.1 Feature-based layout

```text
src/
├── app/
│   ├── App.tsx
│   ├── router.tsx
│   ├── providers.tsx
│   └── styles.css
├── components/
│   ├── ui/                 # primitives: Button, Input, Dialog, Card, Badge, Table
│   └── layout/             # AppShell, Sidebar, Topbar, PageHeader
├── features/
│   ├── orders/
│   │   ├── api.ts
│   │   ├── components/
│   │   │   ├── OrderTable.tsx
│   │   │   ├── OrderStatusBadge.tsx
│   │   │   └── EmptyOrders.tsx
│   │   ├── hooks.ts
│   │   ├── schemas.ts
│   │   └── types.ts
│   └── settings/
├── routes/
│   ├── dashboard.tsx
│   ├── orders.tsx
│   └── settings.tsx
├── lib/
│   ├── cn.ts
│   ├── api.ts
│   └── format.ts
└── styles/
    └── theme.css
```

Rules:

- `components/ui` contains **presentational primitives** with no feature knowledge.
- `features/*` owns its API, hooks, schemas, types and components.
- `routes/*` composes features into pages; it does not contain business logic.
- `lib` is framework-agnostic utilities.
- Avoid barrel files (`index.ts` re-exports) in hot paths; they hurt tree-shaking and make diffs noisy. Import directly.
- One component per file; file name = component name.

### 5.2 Component taxonomy

| Level | Examples | Rules |
|---|---|---|
| **Primitive** | `Button`, `Input`, `Dialog`, `Tabs`, `Tooltip` | headless behaviour + tokens; no domain data |
| **Pattern** | `PageHeader`, `DataTable`, `EmptyState`, `ConfirmDialog` | composable, slot-based; no feature API |
| **Feature** | `OrderTable`, `OrderForm`, `OrderStatusBadge` | knows the domain; uses primitives |
| **Page** | `OrdersPage` | routing, data loading, layout; thin |

Composition over configuration. A `Button` with 30 boolean props is a smell; use variants and children/slots.

### 5.3 State matrix

| State | Tool | Examples |
|---|---|---|
| **Server state** | TanStack Query | lists, details, mutations, cache invalidation |
| **URL state** | React Router `useSearchParams` | filters, pagination, tabs, sort |
| **Form state** | React 19 Actions / `useActionState` | create/edit forms, validation errors |
| **Local UI state** | `useState` / `useReducer` | open/closed, selected row, draft |
| **Global client state** | Context, or Zustand only if needed | theme, session, command palette |
| **Derived state** | compute during render | filtered lists, counts |

Rules:

- Do not copy server data into `useState`; it creates stale duplicates.
- Do not put filters in local state when they should survive reload; use the URL.
- Do not use `useEffect` to derive state from props; compute it.
- Do not use `useEffect` to fetch when TanStack Query or a route loader is available.
- Keep global state small. A 14B model will overuse Context if the data shows it.

### 5.4 Server state with TanStack Query

```tsx
import { useQuery } from "@tanstack/react-query";
import { ordersQuery } from "@/features/orders/api";

export function OrdersPage() {
  const { data, isPending, isError, refetch } = useQuery(ordersQuery.list());
  if (isPending) return <OrdersSkeleton />;
  if (isError) return <ErrorState onRetry={refetch} />;
  if (data.length === 0) return <EmptyOrders />;
  return <OrderTable orders={data} />;
}
```

The component has **four explicit states**: loading, error, empty, content. Training data must show all four; a model trained only on the happy path will ship blank screens.

### 5.5 React 19 patterns

**`ref` as a prop** — no `forwardRef` wrapper:

```tsx
export function Field({ ref, ...props }: FieldProps) {
  return <input ref={ref} {...props} />;
}
```

**Actions and `useActionState`**:

```tsx
const [state, action, pending] = useActionState(updateProfile, initialState);

return (
  <form action={action}>
    <Field name="name" defaultValue={state.values?.name} />
    {state.errors?.name && <FieldError>{state.errors.name}</FieldError>}
    <Button type="submit" disabled={pending}>Save</Button>
  </form>
);
```

**`useOptimistic`** for instant feedback, with rollback on error.

**`use`** for reading a promise or context in a client component, inside Suspense.

**View Transitions** for route/list transitions; always respect `prefers-reduced-motion`.

**React Compiler** is stable (1.0, Oct 2025). If it is enabled, do not sprinkle `useMemo`/`useCallback` everywhere; write plain components and let the compiler memoize. Do not break the Rules of React.

### 5.6 Routing

React Router 8 data mode:

```tsx
import { createBrowserRouter } from "react-router";
import { RouterProvider } from "react-router/dom";

const router = createBrowserRouter([
  { path: "/", element: <AppShell />, children: [
    { index: true, element: <DashboardPage /> },
    { path: "orders", element: <OrdersPage /> },
    { path: "orders/:id", element: <OrderDetailPage /> },
  ]},
]);

export function App() {
  return <RouterProvider router={router} />;
}
```

Use `errorElement` / `ErrorBoundary` for route errors; use `lazy` for route code splitting.

---

## 6. Tailwind 4 as a design system

### 6.1 Tokens, not raw palette

```css
@import "tailwindcss";

@custom-variant dark (&:where(.dark, .dark *));

@theme {
  --font-sans: "Inter Variable", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;

  --color-canvas: oklch(0.985 0.003 250);
  --color-surface: oklch(1 0 0);
  --color-ink: oklch(0.22 0.02 260);
  --color-muted: oklch(0.55 0.02 260);
  --color-border: oklch(0.91 0.01 260);
  --color-brand: oklch(0.58 0.19 275);
  --color-brand-foreground: oklch(0.99 0 0);
  --color-success: oklch(0.52 0.12 155);
  --color-danger: oklch(0.55 0.18 27);

  --radius-card: 1rem;
  --shadow-card: 0 1px 2px rgb(0 0 0 / 0.04), 0 8px 24px rgb(0 0 0 / 0.06);
}

@layer base {
  html { color-scheme: light; }
  .dark { color-scheme: dark; }
  body { @apply bg-canvas text-ink antialiased; }
}

.dark {
  --color-canvas: oklch(0.16 0.01 260);
  --color-surface: oklch(0.20 0.012 260);
  --color-ink: oklch(0.96 0.004 260);
  --color-muted: oklch(0.72 0.015 260);
  --color-border: oklch(0.30 0.015 260);
  --color-success: oklch(0.72 0.12 155);
  --color-danger: oklch(0.70 0.16 27);
}
```

Then components use **semantic** classes:

```tsx
<section className="rounded-card border border-border bg-surface p-6 shadow-card">
  <h2 className="text-lg font-semibold tracking-tight">Recent orders</h2>
  <p className="mt-1 text-sm text-muted">Updated five minutes ago.</p>
</section>
```

Not:

```tsx
<section className="rounded-2xl border border-gray-200 bg-white p-6 shadow-lg">
```

The second is not “wrong”; it is **unmaintainable** and it will not survive dark mode or a rebrand.

### 6.2 Variants with CVA and `cn`

```ts
// src/lib/cn.ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

```tsx
// src/components/ui/button.tsx
import { cva, type VariantProps } from "class-variance-authority";
import type { ComponentProps } from "react";
import { cn } from "@/lib/cn";

const button = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors " +
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand " +
    "disabled:pointer-events-none disabled:opacity-50 motion-reduce:transition-none",
  {
    variants: {
      variant: {
        primary: "bg-brand text-brand-foreground hover:bg-brand/90",
        secondary: "border border-border bg-surface text-ink hover:bg-canvas",
        ghost: "text-ink hover:bg-canvas",
        danger: "bg-red-600 text-white hover:bg-red-700",
      },
      size: {
        sm: "h-9 px-3 text-sm",
        md: "h-10 px-4 text-sm",
        lg: "h-11 px-5 text-base",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

type ButtonProps = ComponentProps<"button"> & VariantProps<typeof button>;

export function Button({ className, variant, size, type = "button", ...props }: ButtonProps) {
  return <button type={type} className={cn(button({ variant, size }), className)} {...props} />;
}
```

Why this shape:

- variants are typed and exhaustive;
- `cn` resolves conflicting Tailwind classes;
- `type="button"` is the default, preventing accidental form submits;
- focus styles are explicit;
- motion respects reduced motion;
- no raw hex in the component.

### 6.3 Rules that prevent broken Tailwind

- **No dynamic class construction**: `bg-${color}-500` is invisible to Tailwind's scanner. Use a map of complete class strings.
- **No arbitrary hex** (`bg-[#6d28d9]`) when a token exists.
- **No `@apply` in component CSS without `@reference`** in v4 CSS modules.
- **No `tailwind.config.js`** for a new v4 project; use `@theme`. Legacy config is only via `@config`.
- **No `@tailwind base`**; use `@import "tailwindcss"`.
- **No `outline-none`** on focusable elements; use `outline-hidden` for the old behaviour or an explicit focus ring.
- **No `bg-opacity-*`**; use `bg-black/50`.
- Use `gap-*` instead of `space-x-*`/`space-y-*` in new flex/grid layouts.
- Use `min-h-dvh`, not `min-h-screen`, for mobile browser chrome.
- Use `text-balance` for headings, `text-pretty` for body copy.
- Use container queries (`@container`, `@container-size`) for reusable components, viewport breakpoints for page layout.

### 6.4 Dark mode

Class strategy:

```css
@custom-variant dark (&:where(.dark, .dark *));
```

Toggle `document.documentElement.classList.toggle("dark")`; persist the choice; respect `prefers-color-scheme` on first load. Do not hardcode `dark:` colors that ignore the token system. Test contrast in both themes.

### 6.5 Responsive and fluid

- Mobile-first; test 320, 375, 768, 1024, 1440.
- Page container: `mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8`.
- Fluid type only where it helps: `text-[clamp(1.75rem,1.2rem+2vw,2.75rem)]`.
- Tables: wrap in `overflow-x-auto`, keep a caption and `scope="col"`.
- Never hide critical actions at small widths; move them into a menu, not into oblivion.
- Check for horizontal overflow at 320px.

---

## 7. Accessibility: WCAG 2.2 AA as a gate

| Requirement | Practical rule |
|---|---|
| **Semantic HTML** | `button` for actions, `a` for navigation, `nav`/`main`/`header`/`footer`, one `h1` |
| **Labels** | every input has a visible `<label>` or `aria-label`; placeholder is not a label |
| **Keyboard** | every action reachable by Tab; Enter/Space activate; Escape closes overlays |
| **Focus** | visible focus ring; not obscured by sticky headers (2.4.11); `scroll-padding-top` |
| **Target size** | at least 24×24 CSS px (2.5.8 AA); 44×44 is a good default |
| **Contrast** | 4.5:1 for normal text, 3:1 for large text and UI boundaries |
| **Images** | meaningful `alt`; decorative `alt=""`; no text baked into images |
| **Forms** | error text linked with `aria-describedby`; `aria-invalid`; summary at top |
| **Motion** | `motion-reduce:` / `prefers-reduced-motion`; no autoplay |
| **Dragging** | provide a non-drag alternative (2.5.7) |
| **Auth** | no cognitive-function test; paste and password managers allowed |

Use **Radix primitives** for dialogs, menus, tabs, tooltips, switches and selects; they handle focus trapping, roving tabindex, typeahead, escape and ARIA. Then style with Tailwind.

`axe` catches maybe 30–50% of issues. Manual keyboard and screen-reader checks remain mandatory.

---

## 8. Performance and “nice” quality

### 8.1 Performance

- Route-level code splitting with `lazy` and Suspense.
- Image dimensions, `srcset`, `loading="lazy"` below the fold, `fetchpriority="high"` for LCP.
- Fonts: `font-display: swap`, preload only the used weight, subset.
- Avoid layout shift: reserve space for async content, skeletons with the same dimensions.
- INP: keep event handlers small; avoid long lists without virtualization; use `startTransition` for non-urgent updates.
- React Compiler can remove unnecessary memoization; measure before adding `useMemo`.
- Bundle analysis: `vite build --mode analyze` or `rollup-plugin-visualizer`; budget per route.
- CSS: Tailwind emits only used classes; avoid huge `@apply` layers.

### 8.2 Visual quality rubric

“Nice” is not a class list. Grade each screen 1–5 on:

1. **Hierarchy** — one primary action per view; scanning order is obvious.
2. **Spacing** — consistent 4/8px rhythm; generous whitespace; aligned edges.
3. **Typography** — one display face, one body face; 1.2–1.333 scale; 45–75 characters per line.
4. **Color** — one accent, neutral canvas, semantic states; contrast holds in dark mode.
5. **Density** — tables and forms are scannable; not cramped, not airy to the point of emptiness.
6. **States** — loading, empty, error, disabled, hover, focus, active, selected.
7. **Responsiveness** — no overflow; navigation adapts; touch targets hold.
8. **Restraint** — motion and decoration serve meaning; no generic “AI” gradient soup.

### 8.3 “AI slop” to reject in training data

- purple-to-pink gradient on every hero;
- glassmorphism cards with no hierarchy;
- 32px radius on everything;
- emoji as the icon system;
- `Lorem ipsum`, fake testimonials, fake logos;
- low-contrast gray text on white;
- centered everything, no grid;
- five different shadows and radii;
- decorative animation that ignores reduced motion;
- a dashboard with only a happy path.

Train the opposite: real copy, one accent, a tokenized palette, consistent radii, subtle elevation, a clear grid, and explicit states.

---

## 9. Testing strategy

| Layer | Tool | What to verify |
|---|---|---|
| **Type** | `tsc --noEmit` (TS 7) | no type errors, no removed options |
| **Lint** | ESLint 10 + react-hooks + jsx-a11y | hooks rules, a11y, imports |
| **Unit/component** | Vitest 5 + Testing Library | behavior via roles, not implementation |
| **Stories** | Storybook 10.6 (optional) | variants, states, a11y addon |
| **E2E** | Playwright 1.64 | critical flows, responsive, screenshots |
| **A11y** | `@axe-core/playwright` | serious/critical violations = 0 |
| **Visual** | Playwright `toHaveScreenshot` | refactors preserve appearance |
| **Build** | `vite build` | production build, no missing imports |

Type-check and build are separate: Vite transpiles without type-checking. An agent that only runs `vite build` can ship type errors.

---

## 10. What not to mix

| Mix | Symptom | Rule |
|---|---|---|
| React 18 + 19 idioms | `forwardRef` everywhere, `ReactDOM.render`, `useFormState` | tag `react: 19` |
| Tailwind 3 + 4 | `@tailwind base` + `@theme`, wrong shadow/radius scale | tag `tailwind: 4` |
| Next.js + Vite SPA | `next/link`, `"use server"`, route handlers in a Vite app | separate track |
| React Router 7 + 8 | `react-router-dom` imports | tag `router: 8` |
| TypeScript 6 + 7 config | `baseUrl`, `moduleResolution: node` | TS 7-compatible tsconfig |
| CSS-in-JS + Tailwind | two styling systems, specificity fights | pick one |
| Redux + TanStack Query | duplicated server state | Query for server, Context/Zustand for UI |

---

## 11. The data plan in one table

| Bucket | Share | Source |
|---|---:|---|
| Components/primitives | 20% | shadcn/ui, Radix, synthetic |
| Pages/layouts | 15% | synthetic, react-code-instructions |
| Data/forms | 15% | TanStack Query docs, synthetic |
| Design system/refactor | 10% | synthetic + tokens |
| Accessibility | 10% | synthetic + axe |
| Testing | 10% | synthetic + Playwright + Testing Library |
| Bug fix/recovery | 10% | commits, synthetic |
| Migration (React 18→19, Tailwind 3→4, Router 7→8, TS 6→7) | 5% | synthetic |
| Negative/refusal/security | 5% | synthetic |

Details, licenses and exact dataset IDs are in [23 — React & Tailwind datasets](23-react-datasets.md). The training runbook is [24 — React fine-tuning](24-react-finetuning.md). The synthetic pipeline is [25 — React synthetic data](25-react-synthetic.md).

---

## 12. Sources

- React 19.3 — <https://react.dev/blog/2026/09/09/react-19-3>
- React versions — <https://react.dev/versions>
- React Compiler 1.0 — <https://react.dev/blog/2025/10/07/react-compiler-1>
- Tailwind CSS v4.3 — <https://tailwindcss.com/blog/tailwindcss-v4-3>
- Tailwind v4 upgrade guide — <https://tailwindcss.com/docs/upgrade-guide>
- Tailwind theme variables — <https://tailwindcss.com/docs/theme>
- Tailwind dark mode — <https://tailwindcss.com/docs/dark-mode>
- Vite 8 — <https://vite.dev/blog/announcing-vite8>
- Vite 8 migration — <https://vite.dev/guide/migration>
- TypeScript 7.0 — <https://devblogs.microsoft.com/typescript/announcing-typescript-7-0>
- TypeScript 7.0 RC (removed options) — <https://devblogs.microsoft.com/typescript/announcing-typescript-7-0-rc>
- React Router v8 — <https://remix.run/blog/react-router-v8>
- React Router changelog — <https://reactrouter.com/changelog>
- Node.js releases — <https://nodejs.org/en/about/previous-releases>
- TanStack Query — <https://tanstack.com/query/latest>
- Radix Primitives — <https://www.radix-ui.com/primitives>
- shadcn/ui — <https://ui.shadcn.com/>
- WCAG 2.2 — <https://www.w3.org/TR/WCAG22/>
- WCAG 2.5.8 target size — <https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html>
- Storybook 10 — <https://storybook.js.org/releases/10.0>
- Playwright — <https://playwright.dev/>
- axe-core — <https://github.com/dequelabs/axe-core>
- Qwen2.5-Coder-14B — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct>
- Qwen3-Coder-30B-A3B — <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>
