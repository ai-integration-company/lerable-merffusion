import { z } from 'zod';

export const StepperFormSchema = z.object({
  productPicture: z.custom<File>(),
  backgroundPicture: z.custom<File>().nullish(),
  backgroundDescription: z.string().nullish(),
  styleTemplate: z.string().nullish(),
  additionalParameters: z.object({ slider1: z.number(), slider2: z.number(), slider3: z.number() }),
});
