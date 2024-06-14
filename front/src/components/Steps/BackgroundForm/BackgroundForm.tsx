// @ts-nocheck
import { Box, Button, Flex, rem, Stack, Tabs, Text, Textarea, Tooltip } from '@mantine/core';
import { BookDashed, CircleHelp, Scan } from 'lucide-react';
import { useForm, FormProvider } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { BackgroundFormData, BackgroundFormProps } from './types';
import { defaultValues, templatesList } from './utils';
import { BackgroundFormSchema } from './schemas';
import { Dropzone } from '@/components/Dropzone';

export const BackgroundForm = ({ onSubmit, actions }: BackgroundFormProps) => {
  const form = useForm<BackgroundFormData>({
    resolver: zodResolver(BackgroundFormSchema),
    defaultValues,
  });
  const { setValue, watch, register, handleSubmit, formState } = form;
  const styleTemplate = watch('styleTemplate');
  const type = watch('type');
  console.log(form.formState.errors);
  return (
    <FormProvider {...form}>
      <form onSubmit={handleSubmit(onSubmit)}>
        <Stack>
          <Tabs value={type} onChange={(value) => setValue('type', value)}>
            <Tabs.List>
              <Tabs.Tab
                value="TEMPLATE"
                leftSection={<BookDashed size={20} />}
                rightSection={
                  <Tooltip label="В качестве фона вы можете выбрать один из шаблонов">
                    <CircleHelp size={16} />
                  </Tooltip>
                }
              >
                Шаблоны
              </Tabs.Tab>
              <Tabs.Tab
                value="CUSTOM"
                leftSection={<Scan size={20} />}
                rightSection={
                  <Tooltip label="Можно самому загрузить фон или описать его">
                    <CircleHelp size={16} />
                  </Tooltip>
                }
              >
                Свой фон
              </Tabs.Tab>
            </Tabs.List>
          </Tabs>
          {type === 'TEMPLATE' && (
            <Flex gap="sm" wrap="wrap" justify="center">
              {templatesList.map((template) => (
                <Button
                  key={template.name}
                  h="fit-content"
                  p={0}
                  variant="white"
                  bg="transparent"
                  color="black"
                  onClick={() => setValue('styleTemplate', template.name)}
                >
                  <Flex direction="column" align="center">
                    <Box
                      w={rem(100)}
                      mih={rem(100)}
                      bg="dark.0"
                      style={{
                        borderRadius: rem(16),
                        border: styleTemplate === template.name ? '1px solid blue' : undefined,
                      }}
                    />
                    <Text>{template.name}</Text>
                  </Flex>
                </Button>
              ))}
              {formState?.errors.styleTemplate && (
                <Text size="xs" color="red">
                  {formState?.errors.styleTemplate.message}
                </Text>
              )}
            </Flex>
          )}
          {type === 'CUSTOM' && (
            <Flex direction="column" align="center" gap="md">
              <Textarea
                rows={3}
                label="Описание фона"
                {...register('backgroundDescription')}
                style={{ flexGrow: 1, width: '100%', maxWidth: 540 }}
                radius={rem(12)}
              />
              {formState?.errors.backgroundDescription && (
                <Text size="xs" color="red">
                  {formState?.errors.backgroundDescription.message}
                </Text>
              )}
              <Dropzone<BackgroundFormData> name="backgroundPicture" />
            </Flex>
          )}
          {actions}
        </Stack>
      </form>
    </FormProvider>
  );
};
