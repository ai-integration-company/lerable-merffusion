import { ReactNode } from 'react';
import { SubmitHandler } from 'react-hook-form';
import { z } from 'zod';
import { ParametersFormSchema } from './schemas';

export type ProductFormData = z.infer<typeof ParametersFormSchema>;

export type ParametersFormProps = {
  onSubmit: SubmitHandler<ProductFormData>;
  actions: ReactNode;
};
