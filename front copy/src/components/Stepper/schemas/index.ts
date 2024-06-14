import { z } from 'zod';

export const StepperFormSchema = z.object({
  productPicture: z.custom<File>(),
});
