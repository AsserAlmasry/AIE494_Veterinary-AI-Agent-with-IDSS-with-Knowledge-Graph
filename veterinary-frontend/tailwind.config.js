/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary brand: deep forest green (trust, nature, health)
        brand: {
          50:  '#f0faf4',
          100: '#dcf4e6',
          200: '#b8e8ce',
          300: '#85d3aa',
          400: '#4cb87f',
          500: '#2a9d5c',
          600: '#1d7f49',
          700: '#16663a',
          800: '#125230',
          900: '#0d3f24',
          950: '#072518',
        },
        // Secondary: warm sage
        sage: {
          50:  '#f5f8f2',
          100: '#e8f0e2',
          200: '#d0e1c5',
          300: '#afcca0',
          400: '#87b276',
          500: '#669757',
          600: '#507944',
          700: '#3f6035',
          800: '#354e2d',
          900: '#2c4126',
        },
        // Accent: amber for alerts / warnings
        alert: {
          green:  '#16a34a',
          amber:  '#d97706',
          red:    '#dc2626',
          blue:   '#2563eb',
        },
        // Clinical neutral
        clinical: {
          50:  '#f8fafb',
          100: '#f1f5f5',
          200: '#e2edec',
          300: '#c5d8d6',
          400: '#93b6b3',
          500: '#5d8f8b',
          600: '#3d6e6a',
          700: '#2d5450',
          800: '#1f3b38',
          900: '#132724',
        },
        health: {
          emerald: '#10b981',
          amber:   '#f59e0b',
          rose:    '#f43f5e',
          sky:     '#0ea5e9',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Outfit', 'Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow':    'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float':         'float 6s ease-in-out infinite',
        'float-delayed': 'float 6s ease-in-out 2s infinite',
        'float-slow':    'float 8s ease-in-out 1s infinite',
        'fade-in':       'fadeIn 0.5s ease-out',
        'slide-up':      'slideUp 0.4s ease-out',
        'cow-walk':      'cowWalk 8s linear infinite',
        'glow':          'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%':      { transform: 'translateY(-12px)' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        cowWalk: {
          '0%':   { transform: 'translateX(-200px)' },
          '100%': { transform: 'translateX(calc(100vw + 200px))' },
        },
        glow: {
          from: { boxShadow: '0 0 5px rgba(42, 157, 92, 0.3)' },
          to:   { boxShadow: '0 0 20px rgba(42, 157, 92, 0.7)' },
        },
      },
      boxShadow: {
        'card':     '0 2px 16px rgba(13, 63, 36, 0.06), 0 1px 4px rgba(0,0,0,0.04)',
        'card-lg':  '0 8px 32px rgba(13, 63, 36, 0.10), 0 2px 8px rgba(0,0,0,0.06)',
        'brand':    '0 4px 20px rgba(42, 157, 92, 0.3)',
        'inner-sm': 'inset 0 2px 4px rgba(0,0,0,0.06)',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}
