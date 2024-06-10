'use client';

import { createTheme, rem } from '@mantine/core';

export const InputSizes = {
  xs: rem(36),
  sm: rem(42),
  md: rem(44),
  lg: rem(50),
  xl: rem(60),
};

export const theme = createTheme({
  //@ts-ignore
  colorScheme: 'light',
  fontFamily: 'Roboto, sans-serif',
  colors: {
    purple: [
      '#EDF2FF',
      '#DBE4FF',
      '#BAC8FF',
      '#A7B0DE',
      '#8592DE',
      '#5E72E4',
      '#5265D0',
      '#4A5BBB',
      '#4B57A0',
      '#4B5488',
    ],
    cyan: [
      '#e9fcfa',
      '#dcf2f2',
      '#bae3e3',
      '#95d4d4',
      '#76c8c7',
      '#62c0bf',
      '#55bcbb',
      '#44a5a4',
      '#369392',
      '#1e807f',
    ],
  },

  black: '#000',
  white: '#fff',

  cursorType: 'pointer',

  radius: {
    xs: '0.125rem',
    sm: '0.25rem',
    md: '0.375rem',
    lg: '0.5rem',
  },

  breakpoints: {
    '2xl': '96em',
  },

  primaryColor: 'purple',
  primaryShade: 6,

  spacing: {
    xxl: rem(40),
    xl: rem(28),
  },

  components: {
    Button: {
      defaultProps: {
        radius: 'md',
      },
    },

    Tooltip: {
      defaultProps: {
        withArrow: true,
        withinPortal: true,
        arrowSize: 6,
      },
    },
    Input: {
      defaultProps: {
        radius: rem(16),
        size: 'sm',
      },
    },

    Stepper: {
      defaultProps: {
        size: 'sm',
      },
    },
    Modal: {
      styles: {
        title: {
          fontWeight: 600,
        },
      },
    },
  },
});
