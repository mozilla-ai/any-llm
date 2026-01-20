pnpm install
pnpm exec prettier --write .
pnpm exec eslint . --fix
pnpm exec tsc -b --noEmit
