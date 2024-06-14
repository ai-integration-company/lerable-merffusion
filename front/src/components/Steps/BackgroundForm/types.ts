import { z } from 'zod';

import { SubmitHandler } from 'react-hook-form';
import { ReactNode } from 'react';
import { BackgroundFormSchema } from './schemas';

export type TabsVariants = { TEMPLATE: 'template'; CUSTOM: 'custom' };

export type BackgroundFormData = z.infer<typeof BackgroundFormSchema>;

export type BackgroundFormProps = {
  onSubmit: SubmitHandler<BackgroundFormData>;
  actions: ReactNode;
};
