# Smart Attendance System - Theme Customization Guide

**Version 2.0** | Last Updated: November 2024

## Overview

This guide explains how to customize the visual appearance of the Smart Attendance System by modifying design tokens (CSS variables). You don't need to be a programmer, but basic understanding of colors and CSS will be helpful.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Design Tokens](#understanding-design-tokens)
3. [Customizing Colors](#customizing-colors)
4. [Customizing Spacing](#customizing-spacing)
5. [Customizing Typography](#customizing-typography)
6. [Creating Custom Themes](#creating-custom-themes)
7. [Examples](#examples)

---

## Quick Start

### Where to Make Changes

All design tokens are defined in one file:
```
recognition/static/css/app.css
```

Open this file in a text editor to make changes.

### Basic Steps

1. Open `recognition/static/css/app.css`
2. Find the `:root {` section at the top
3. Modify the values you want to change
4. Save the file
5. Refresh your browser (you may need to do a hard refresh: Ctrl+F5 or Cmd+Shift+R)

---

## Understanding Design Tokens

Design tokens are named variables that store design values. Instead of using colors, sizes, and spacing directly in your code, you use these token names. This makes it easy to change the entire design by modifying one file.

### Example

Instead of:
```css
color: #0d6efd;  /* What does this color mean? */
```

We use:
```css
color: var(--color-primary);  /* Clearly a primary color */
```

Now, changing `--color-primary` in one place updates it everywhere in the application.

---

## Customizing Colors

### Finding Color Tokens

In `app.css`, find the `:root {` section. You'll see color definitions like:

```css
:root {
  /* Primary colors */
  --color-primary: #0d6efd;
  --color-primary-dark: #0b5ed7;
  --color-primary-light: #6ea8fe;
  
  /* Semantic colors */
  --color-success: #198754;
  --color-danger: #dc3545;
  --color-warning: #ffc107;
  --color-info: #0dcaf0;
  
  /* ... more colors */
}
```

### Color Token Reference

| Token | Default | Used For |
|-------|---------|----------|
| `--color-primary` | #0d6efd (Blue) | Main brand color, buttons, links |
| `--color-primary-dark` | #0b5ed7 (Dark Blue) | Button hover states |
| `--color-primary-light` | #6ea8fe (Light Blue) | Highlights, backgrounds |
| `--color-success` | #198754 (Green) | Success messages, positive indicators |
| `--color-danger` | #dc3545 (Red) | Error messages, warnings |
| `--color-warning` | #ffc107 (Yellow) | Warning messages |
| `--color-info` | #0dcaf0 (Cyan) | Information messages |
| `--color-background` | #ffffff (White) | Page background |
| `--color-surface` | #ffffff (White) | Card backgrounds |
| `--color-text` | #212529 (Dark Gray) | Main text color |
| `--color-text-muted` | #6c757d (Gray) | Secondary text, labels |
| `--color-border` | #ced4da (Light Gray) | Borders, dividers |

### Changing the Primary Color

To change the main brand color (affects buttons, links, icons):

1. Open `app.css`
2. Find `--color-primary: #0d6efd;`
3. Change to your desired color (use hex, rgb, or color name)
4. Update related colors for consistency

Example - Change to Purple:

```css
:root {
  --color-primary: #6f42c1;           /* Purple */
  --color-primary-dark: #5936a3;      /* Darker purple for hover */
  --color-primary-light: #9775d8;     /* Lighter purple for highlights */
}
```

Example - Change to Green:

```css
:root {
  --color-primary: #28a745;           /* Green */
  --color-primary-dark: #218838;      /* Darker green */
  --color-primary-light: #4cbb6c;     /* Lighter green */
}
```

### Color Formats

You can use different color formats:

```css
/* Hexadecimal (most common) */
--color-primary: #0d6efd;

/* RGB */
--color-primary: rgb(13, 110, 253);

/* RGBA (with transparency) */
--color-primary: rgba(13, 110, 253, 0.9);

/* HSL (Hue, Saturation, Lightness) */
--color-primary: hsl(213, 97%, 52%);

/* Color names */
--color-primary: blue;  /* Not recommended for branding */
```

### Tools for Choosing Colors

- [Coolors](https://coolors.co/) - Color palette generator
- [Adobe Color](https://color.adobe.com/) - Color wheel and schemes
- [Material Design Colors](https://materialui.co/colors) - Pre-made palettes
- [Contrast Checker](https://webaim.org/resources/contrastchecker/) - Ensure readability

### Customizing Dark Mode Colors

Dark mode colors are defined separately:

```css
.theme-dark {
  --color-primary: #6ea8fe;
  --color-background: #0d1117;
  --color-surface: #161b22;
  --color-text: #e6edf3;
  --color-text-muted: #8b949e;
  --color-border: #30363d;
  /* ... */
}
```

Modify these values to customize the dark theme appearance.

---

## Customizing Spacing

Spacing tokens control margins, padding, and gaps between elements.

### Spacing Scale

```css
:root {
  --space-1: 0.25rem;  /* 4px */
  --space-2: 0.5rem;   /* 8px */
  --space-3: 0.75rem;  /* 12px */
  --space-4: 1rem;     /* 16px */
  --space-5: 1.25rem;  /* 20px */
  --space-6: 1.5rem;   /* 24px */
  --space-8: 2rem;     /* 32px */
  --space-10: 2.5rem;  /* 40px */
  --space-12: 3rem;    /* 48px */
  --space-16: 4rem;    /* 64px */
}
```

### Making Spacing More Compact

To make the design more compact (less whitespace):

```css
:root {
  --space-1: 0.2rem;   /* Reduced from 0.25rem */
  --space-2: 0.4rem;   /* Reduced from 0.5rem */
  --space-3: 0.6rem;   /* Reduced from 0.75rem */
  --space-4: 0.8rem;   /* Reduced from 1rem */
  /* ... continue pattern */
}
```

### Making Spacing More Generous

To add more whitespace:

```css
:root {
  --space-1: 0.3rem;   /* Increased from 0.25rem */
  --space-2: 0.6rem;   /* Increased from 0.5rem */
  --space-3: 0.9rem;   /* Increased from 0.75rem */
  --space-4: 1.2rem;   /* Increased from 1rem */
  /* ... continue pattern */
}
```

---

## Customizing Typography

### Font Family

Change the default font:

```css
:root {
  --font-family-base: 'Your Font Name', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
```

**Important**: If using a web font (like Google Fonts), add the font link to `base.html`:

```html
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap" rel="stylesheet">
```

Then update the CSS:

```css
:root {
  --font-family-base: 'Roboto', sans-serif;
}
```

### Font Sizes

```css
:root {
  --font-size-xs: 0.75rem;    /* 12px */
  --font-size-sm: 0.875rem;   /* 14px */
  --font-size-base: 1rem;     /* 16px */
  --font-size-lg: 1.125rem;   /* 18px */
  --font-size-xl: 1.25rem;    /* 20px */
  --font-size-2xl: 1.5rem;    /* 24px */
  --font-size-3xl: 1.875rem;  /* 30px */
  --font-size-4xl: 2.25rem;   /* 36px */
}
```

#### Making Text Larger (Accessibility)

```css
:root {
  --font-size-base: 1.125rem;  /* Increased from 1rem (18px instead of 16px) */
  /* Adjust others proportionally */
  --font-size-lg: 1.25rem;
  --font-size-xl: 1.5rem;
  /* ... */
}
```

#### Making Text Smaller

```css
:root {
  --font-size-base: 0.875rem;  /* Reduced from 1rem (14px instead of 16px) */
  /* Adjust others proportionally */
  --font-size-lg: 1rem;
  --font-size-xl: 1.125rem;
  /* ... */
}
```

### Font Weights

```css
:root {
  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 600;
  --font-weight-bold: 700;
}
```

### Line Heights

```css
:root {
  --line-height-tight: 1.25;
  --line-height-normal: 1.5;
  --line-height-relaxed: 1.75;
}
```

---

## Creating Custom Themes

You can create multiple themes and allow users to switch between them.

### Step 1: Define Theme Class

Add your theme after the dark theme definition:

```css
.theme-blue {
  --color-primary: #0d6efd;
  --color-primary-dark: #0b5ed7;
  --color-primary-light: #6ea8fe;
}

.theme-green {
  --color-primary: #28a745;
  --color-primary-dark: #218838;
  --color-primary-light: #4cbb6c;
}

.theme-purple {
  --color-primary: #6f42c1;
  --color-primary-dark: #5936a3;
  --color-primary-light: #9775d8;
}
```

### Step 2: Apply Theme

Add the theme class to the `<html>` element:

```html
<html lang="en" class="theme-blue">
```

Or switch programmatically with JavaScript:

```javascript
document.documentElement.className = 'theme-green';
```

---

## Examples

### Example 1: Corporate Blue Theme

```css
:root {
  /* Brand colors */
  --color-primary: #005eb8;           /* Corporate blue */
  --color-primary-dark: #003d7a;      /* Darker blue */
  --color-primary-light: #338fd1;     /* Lighter blue */
  
  /* Keep other colors standard */
  --color-success: #28a745;
  --color-danger: #dc3545;
  --color-warning: #ffc107;
  --color-info: #17a2b8;
}
```

### Example 2: Warm Orange Theme

```css
:root {
  --color-primary: #ff6f00;           /* Warm orange */
  --color-primary-dark: #c43e00;      /* Darker orange */
  --color-primary-light: #ff9e40;     /* Lighter orange */
  
  --color-success: #4caf50;
  --color-danger: #f44336;
  --color-warning: #ffeb3b;
  --color-info: #00bcd4;
}
```

### Example 3: Minimalist Grayscale

```css
:root {
  --color-primary: #2c3e50;           /* Dark blue-gray */
  --color-primary-dark: #1a252f;      /* Darker */
  --color-primary-light: #34495e;     /* Lighter */
  
  --color-success: #27ae60;
  --color-danger: #e74c3c;
  --color-warning: #f39c12;
  --color-info: #3498db;
  
  /* Reduce color intensity */
  --color-background: #f5f5f5;
  --color-surface: #ffffff;
}
```

### Example 4: High Contrast (Accessibility)

```css
:root {
  --color-primary: #0056b3;
  --color-primary-dark: #003d82;
  --color-primary-light: #1976d2;
  
  --color-text: #000000;              /* Pure black */
  --color-text-muted: #333333;        /* Dark gray */
  --color-background: #ffffff;         /* Pure white */
  
  /* Increase border visibility */
  --color-border: #666666;            /* Darker border */
}
```

### Example 5: Compact Mobile-Friendly

```css
:root {
  /* Reduce spacing */
  --space-4: 0.75rem;
  --space-6: 1rem;
  --space-8: 1.5rem;
  
  /* Slightly smaller fonts on mobile */
  --font-size-base: 0.9375rem;        /* 15px */
  --font-size-lg: 1.0625rem;          /* 17px */
}

@media (min-width: 768px) {
  :root {
    /* Normal spacing on desktop */
    --space-4: 1rem;
    --space-6: 1.5rem;
    --space-8: 2rem;
    
    /* Normal fonts on desktop */
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
  }
}
```

---

## Border Radius

Control the roundness of corners:

```css
:root {
  --radius-sm: 0.25rem;   /* 4px - Small rounded */
  --radius-md: 0.375rem;  /* 6px - Medium rounded */
  --radius-lg: 0.5rem;    /* 8px - Large rounded */
  --radius-xl: 0.75rem;   /* 12px - Extra large rounded */
  --radius-2xl: 1rem;     /* 16px - Very rounded */
  --radius-full: 9999px;  /* Fully rounded (pills) */
}
```

### Sharp Corners (Material Design Style)

```css
:root {
  --radius-sm: 0;
  --radius-md: 0;
  --radius-lg: 0;
  --radius-xl: 2px;    /* Minimal radius */
  --radius-2xl: 4px;
}
```

### Very Rounded (iOS Style)

```css
:root {
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 20px;
  --radius-2xl: 24px;
}
```

---

## Shadows

Control shadow depth:

```css
:root {
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}
```

### Flat Design (No Shadows)

```css
:root {
  --shadow-sm: none;
  --shadow-md: none;
  --shadow-lg: none;
  --shadow-xl: none;
}
```

### Dramatic Shadows

```css
:root {
  --shadow-sm: 0 2px 4px 0 rgba(0, 0, 0, 0.1);
  --shadow-md: 0 8px 16px -2px rgba(0, 0, 0, 0.15);
  --shadow-lg: 0 20px 30px -6px rgba(0, 0, 0, 0.2);
  --shadow-xl: 0 40px 50px -10px rgba(0, 0, 0, 0.25);
}
```

---

## Testing Your Changes

### Browser Cache

After making changes, you may need to clear your browser cache:

**Hard Refresh:**
- **Windows/Linux**: Ctrl + F5 or Ctrl + Shift + R
- **Mac**: Cmd + Shift + R

### Multiple Browsers

Test your theme in different browsers:
- Chrome
- Firefox
- Safari
- Edge

### Dark Mode

Test both light and dark themes:
1. Toggle dark mode with the moon/sun icon
2. Check that colors still have good contrast
3. Ensure text is readable

### Mobile Devices

Test on mobile:
- Use browser DevTools responsive mode
- Test on actual mobile devices if possible
- Check that spacing and fonts are appropriate

---

## Common Issues

### Colors Not Changing

**Problem**: Changed colors but nothing updates

**Solutions**:
1. Hard refresh your browser (Ctrl+F5 or Cmd+Shift+R)
2. Check that you're editing the correct file
3. Make sure there are no syntax errors (missing semicolons, brackets)
4. Check browser console for CSS errors

### Text Hard to Read

**Problem**: Text contrast is poor

**Solutions**:
1. Use a [contrast checker](https://webaim.org/resources/contrastchecker/)
2. Aim for at least 4.5:1 ratio for normal text
3. Adjust text color or background color
4. Test in dark mode too

### Layout Looks Broken

**Problem**: Spacing or sizing is off

**Solutions**:
1. Make sure all units are consistent (rem, px)
2. Check that you're using valid CSS values
3. Test on different screen sizes
4. Revert changes and make smaller adjustments

---

## Advanced Customization

### Gradients

Create gradient backgrounds:

```css
.custom-gradient {
  background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%);
}
```

### Animations

Customize animation speeds:

```css
:root {
  --transition-fast: 150ms ease;
  --transition-base: 200ms ease;
  --transition-slow: 300ms ease;
}
```

For faster animations:
```css
:root {
  --transition-fast: 100ms ease;
  --transition-base: 150ms ease;
  --transition-slow: 200ms ease;
}
```

For slower, more dramatic animations:
```css
:root {
  --transition-fast: 200ms ease;
  --transition-base: 300ms ease;
  --transition-slow: 500ms ease;
}
```

---

## Backup and Restore

### Before Making Changes

Always back up the original `app.css` file:

```bash
cp recognition/static/css/app.css recognition/static/css/app.css.backup
```

### Restoring Original

If something goes wrong:

```bash
cp recognition/static/css/app.css.backup recognition/static/css/app.css
```

---

## Need Help?

- **User Guide**: See [user-guide.md](user-guide.md) for general usage
- **Developer Guide**: See [developer-guide.md](developer-guide.md) for technical details
- **Color Theory**: [Learn about color theory](https://www.interaction-design.org/literature/topics/color-theory)
- **Accessibility**: [WCAG Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

---

## Summary Checklist

When customizing your theme:

- [ ] Backup original `app.css` file
- [ ] Make changes to design tokens in `:root` section
- [ ] Test in both light and dark modes
- [ ] Check color contrast for accessibility
- [ ] Test on multiple browsers
- [ ] Test on mobile devices
- [ ] Hard refresh browser after changes
- [ ] Document your custom color scheme

---

*Last updated: November 2024 | Version 2.0*
