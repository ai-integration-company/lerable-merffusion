import { FormProvider, useForm } from 'react-hook-form';
import { Flex, Slider, Stack, Text, Tooltip } from '@mantine/core';
import { CircleHelp } from 'lucide-react';
import { ParametersFormProps, ProductFormData } from './types';

export const ParametersForm = ({ onSubmit, actions }: ParametersFormProps) => {
  const form = useForm<ParametersFormData>({ defaultValues: { param1: 1, param2: 1, param3: 1 } });
  const { handleSubmit, register } = form;
  return (
    <FormProvider {...form}>
      <form onSubmit={handleSubmit(onSubmit)}>
        <Stack mt="xl">
          <Flex direction="column" gap="xs">
            <Flex gap="xs" align="center">
              <Text inline>Label 1</Text>
              <Tooltip label="Tooltip 1">
                <CircleHelp size={16} />
              </Tooltip>
            </Flex>
            <Slider max={1} step={0.01} {...register('param1')} />
          </Flex>
          <Flex direction="column" gap="xs">
            <Flex gap="xs" align="center">
              <Text inline>Label 2</Text>
              <Tooltip label="Tooltip 2">
                <CircleHelp size={16} />
              </Tooltip>
            </Flex>
            <Slider max={1} step={0.01} {...register('param2')} />
          </Flex>
          <Flex direction="column" gap="xs">
            <Flex gap="xs" align="center">
              <Text inline>Label 3</Text>
              <Tooltip label="Tooltip 3">
                <CircleHelp size={16} />
              </Tooltip>
            </Flex>
            <Slider max={1} step={0.01} {...register('param3')} />
          </Flex>
          {actions}
        </Stack>
      </form>
    </FormProvider>
  );
};
