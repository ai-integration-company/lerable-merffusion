import { Stack, Text } from '@mantine/core';
import { FormProvider, useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
//@ts-ignore
import { Dropzone } from '@/components/Dropzone';
import { ProductFormData, ProductFormProps } from './types';
import { ProductFormSchema } from './schemas';
import { useStepperForm } from '@/hooks';

export const ProductForm = ({ onSubmit, actions }: ProductFormProps) => {
  const { stepperForm } = useStepperForm();

  const form = useForm<ProductFormData>({
    resolver: zodResolver(ProductFormSchema),
    defaultValues: {
      productPicture: stepperForm ? stepperForm?.productPicture : undefined,
    },
  });
  const {
    handleSubmit,
    formState: { errors },
  } = form;
  console.log(errors);

  return (
    <FormProvider {...form}>
      <form onSubmit={handleSubmit(onSubmit)}>
        <Stack my="lg" p="xl" align="center">
          <Dropzone<ProductFormData> name="productPicture" />
          <Text>
            Сервис для генерации фона для продуктов компании Leroy Merlin с возможностью
            редактирования
          </Text>
        </Stack>
        {actions}
      </form>
    </FormProvider>
  );
};
