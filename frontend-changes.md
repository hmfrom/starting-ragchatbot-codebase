# Frontend Changes: Theme Toggle Button

## Summary
Added a dark/light mode toggle button to the Course Materials Assistant UI.

## Files Modified

### `frontend/index.html`
- Added theme toggle button with sun and moon SVG icons positioned in top-right corner
- Button placed at the top of the body element, outside the main container for fixed positioning
- Updated CSS and JS cache-busting version numbers

### `frontend/style.css`
- Added light theme CSS variables under `[data-theme="light"]` selector
- Added `--code-bg` variable for consistent code block styling across themes
- Updated code block styles to use the new CSS variable
- Added `.theme-toggle` button styles:
  - Fixed position in top-right corner
  - 44px circular button with border and shadow
  - Hover, focus, and active states
  - Sun/moon icon visibility toggling based on theme
  - Rotation animation when toggling
  - Responsive adjustments for mobile

### `frontend/script.js`
- Added `themeToggle` DOM element reference
- Added event listener for theme toggle click
- Added three new functions:
  - `initializeTheme()`: Loads saved theme preference from localStorage on page load
  - `toggleTheme()`: Switches between dark and light themes with animation
  - `setTheme(theme)`: Applies theme and saves preference to localStorage

## Features
- **Icon-based design**: Sun icon shown in dark mode (click to switch to light), moon icon shown in light mode (click to switch to dark)
- **Position**: Fixed position in top-right corner (1rem from edges)
- **Smooth transitions**: 0.3s ease transitions on all theme-related colors and icon rotation animation
- **Persistence**: Theme preference saved to localStorage and restored on page reload
- **Accessibility**:
  - `aria-label` attribute updates dynamically based on current theme
  - `title` attribute provides tooltip
  - Keyboard navigable (focusable with visible focus ring)
  - Button uses semantic HTML

## Light Theme Colors
| Property | Dark | Light |
|----------|------|-------|
| Background | #0f172a | #f8fafc |
| Surface | #1e293b | #ffffff |
| Surface Hover | #334155 | #f1f5f9 |
| Text Primary | #f1f5f9 | #1e293b |
| Text Secondary | #94a3b8 | #64748b |
| Border Color | #334155 | #e2e8f0 |
| Assistant Message | #374151 | #f1f5f9 |
