import { Button, Flex } from '@mantine/core';
import { ActionsProps } from './types';

export const Actions = ({ getBack, backButtonDisabled }: ActionsProps) => (
  <Flex justify="space-between">
    <Button size="md" onClick={getBack} disabled={backButtonDisabled} variant="outline">
      Назад
    </Button>
    <Button type="submit" size="md">
      Далее
    </Button>
  </Flex>
);
