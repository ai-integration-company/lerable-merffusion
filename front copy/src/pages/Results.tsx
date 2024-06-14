import { Flex, Text } from '@mantine/core';
import { Result, Stepper } from '@/components';
import { useWidth } from '@/hooks';

export const ResultsPage = () => {
  const width = useWidth();
  return (
    <Stepper>
      <Flex align="center" justify="center" px={0} mx={0} miw="100%" pos="relative">
        {width >= 1080 ? (
          <Result />
        ) : (
          <Text p="xl" size="xl">
            К сожалению мобильная версия пока недоступна
          </Text>
        )}
      </Flex>
    </Stepper>
  );
};
