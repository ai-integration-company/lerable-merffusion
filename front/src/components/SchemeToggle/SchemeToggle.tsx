'use client';

import { Switch, useMantineColorScheme, useMantineTheme } from '@mantine/core';
import { Moon, Sun } from 'lucide-react';

export const SchemeToggle = () => {
  const { colorScheme, toggleColorScheme } = useMantineColorScheme();
  const theme = useMantineTheme();
  return (
    <Switch
      checked={colorScheme === 'dark'}
      offLabel={<Sun color={theme.colors.yellow[9]} />}
      onLabel={<Moon color={theme.colors.gray[6]} />}
      onChange={() => toggleColorScheme()}
      size="lg"
      color="dark.4"
      pos="absolute"
      mr="xl"
    />
  );
};
