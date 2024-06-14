import { z } from 'zod';
import { SubmitHandler } from 'react-hook-form';
import { ReactNode } from 'react';
import { ProductFormSchema } from './schemas';

export type ProductFormData = z.infer<typeof ProductFormSchema>;

export type ProductFormProps = {
  onSubmit: SubmitHandler<ProductFormData>;
  actions: ReactNode;
};
