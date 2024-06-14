import { Stack, Title } from '@mantine/core';
import { MovingBlockCanvas } from './MovingBlockCanvas';

export const Result = () => (
  <Stack mt="lg">
    <Title order={4}>Результаты генерации:</Title>
    <MovingBlockCanvas />
  </Stack>
);
