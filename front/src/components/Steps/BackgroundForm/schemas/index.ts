import { z } from 'zod';
// import { TabsVariants } from '../types';

export const TabsVariants = ['TEMPLATE', 'CUSTOM'] as const;
export const BackgroundFormSchema = z.discriminatedUnion('type', [
  z.object({
    styleTemplate: z.string().min(1, { message: 'Выберите шаблон' }),
    type: z.literal(TabsVariants[0]),
  }),
  z.object({
    backgroundPicture: z.object({
      fileData: z.string(),
      fileName: z.string(),
      mimeType: z.string(),
    }),
    backgroundDescription: z.string().min(1, { message: 'Необходимо описать фон' }),
    type: z.literal(TabsVariants[1]),
  }),
]);
