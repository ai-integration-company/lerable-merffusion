// @ts-nocheck

'use client';

import { Group, rem, Button, Image, Text } from '@mantine/core';
import { Upload, X, ImageIcon } from 'lucide-react';
import { Controller, useFormContext } from 'react-hook-form';
import { Dropzone as MantineDropzone } from '@mantine/dropzone';
import { useState, useEffect } from 'react';
import { notifications } from '@mantine/notifications';
import { DropzoneProps } from './types';
import { fileToBase64, base64ToFile } from '../../utils';
import { useStepperForm } from '@/hooks';

export const Dropzone = <T,>({ name }: DropzoneProps) => {
  // eslint-disable-next-line
  //@ts-ignore
  const { control, setValue, formState, clearErrors } = useFormContext<T>();
  const { stepperForm, setStepperForm } = useStepperForm();
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  // eslint-disable-next-line consistent-return
  useEffect(() => {
    //@ts-ignore
    // eslint-disable-next-line consistent-return
    const updateImageUrl = async () => {
      //@ts-ignore
      if (stepperForm[name]) {
        const file = base64ToFile(
          // eslint-disable-next-line
          //@ts-ignore
          stepperForm[name].fileData,
          //@ts-ignore
          stepperForm[name].fileName,
          //@ts-ignore
          stepperForm[name].mimeType
        );
        const url = URL.createObjectURL(file);
        setImageUrl(url);
        // eslint-disable-next-line
        //@ts-ignore
        clearErrors([name].fileData);
        return () => URL.revokeObjectURL(url);
      }
      setImageUrl(null);
    };

    updateImageUrl();
  }, [stepperForm]);
  return (
    <Controller
      name={name}
      // eslint-disable-next-line
      //@ts-ignore
      control={control}
      render={({ field }) => (
        <>
          <MantineDropzone
            onDrop={async (files) => {
              const base64 = await fileToBase64(files[0]);
              field.onChange({
                fileData: base64,
                fileName: files[0].name,
                mimeType: files[0].type,
              });
              setStepperForm({
                productPicture: {
                  fileData: base64,
                  fileName: files[0].name,
                  mimeType: files[0].type,
                },
              });
              notifications.show({
                color: 'green',
                title: 'Файл успешно загружен',
                message: '',
              });
            }}
            onReject={() => {
              notifications.show({
                color: 'red',
                title: 'Файл не подходит',
                message: 'Попробуйте другое изображение',
              });
            }}
            style={(theme) => ({
              border: `1px dashed ${theme.colors.gray[4]}`,
              borderRadius: rem(16),
              maxWidth: 600,
              input: {},
              '&[data-idle]': {
                display: 'flex',
                alignItems: 'center',
              },
            })}
            p="md"
            accept={['image/png', 'image/jpeg', 'image/sgv+xml', 'image/gif']}
            // eslint-disable-next-line
            //@ts-ignore
            disabled={stepperForm[name]}
          >
            <Group justify="center" gap="xl" mih={220} style={{ pointerEvents: 'none' }}>
              <MantineDropzone.Accept>
                <Upload
                  style={{
                    width: rem(52),
                    height: rem(52),
                    color: 'var(--mantine-color-blue-6)',
                  }}
                  strokeWidth={1.5}
                />
              </MantineDropzone.Accept>
              <MantineDropzone.Reject>
                <X
                  style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-red-6)' }}
                  strokeWidth={1.5}
                />
              </MantineDropzone.Reject>
              <MantineDropzone.Idle>
                {imageUrl ? (
                  <>
                    <Image
                      src={imageUrl}
                      alt="Uploaded product"
                      mt="md"
                      radius={rem(16)}
                      fit="contain"
                      style={{ maxWidth: '100%', maxHeight: 220, objectFit: 'contain' }}
                    />
                  </>
                ) : (
                  <ImageIcon
                    style={{
                      width: rem(52),
                      height: rem(52),
                      color: 'var(--mantine-color-dimmed)',
                    }}
                    strokeWidth={1.5}
                  />
                )}
              </MantineDropzone.Idle>
              {!imageUrl && (
                <Text size="xs" inline c="dark">
                  Перетащите изображения товара сюда или нажмите, чтобы выбрать файл
                </Text>
              )}
            </Group>
          </MantineDropzone>
          {/* eslint-disable-next-line */}
          {/* @ts-ignore */}
          {formState?.errors[name] && (
            <Text size="xs" color="red">
              {/* eslint-disable-next-line */}
              {/* @ts-ignore */}
              {formState?.errors[name].message === 'Required' ||
              formState?.errors[name].message === 'Expected object, received array'
                ? 'Необходимо загрузить фотографию товара'
                : formState?.errors[name].message}
            </Text>
          )}
          {imageUrl && (
            //@ts-ignore
            <Button
              color="red"
              onClick={() => {
                setValue(name, undefined);
                setStepperForm({ productPicture: undefined });
              }}
              variant="outline"
            >
              Удалить картинку товара
            </Button>
          )}
        </>
      )}
    />
  );
};
