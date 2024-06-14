import axios from 'axios';
import { ApiResponse } from '@/types';

export const getBackground = async (imgString: string) => {
  try {
    const response = await axios.post(
      'https://node-api.datasphere.yandexcloud.net/process_image',
      {
        image: imgString.replace('data:image/png;base64', ''),
        positive_prompt:
          'a background that does not attract much attention, in which the object is clearly visible',
        negative_prompt: 'new objects or entities',
      },
      {
        headers: {
          Authorization:
            'Bearer t1.9euelZqbismZmMeSjpiQlM6JyYqPk-3rnpWaycfMjY2bjceZko_OlImay83l9PcMeE5M-e8GKzSd3fT3TCZMTPnvBis0nc3n9euelZqYjIuXjZbGmpONx5CUi8aKi-_8xeuelZqYjIuXjZbGmpONx5CUi8aKiw.vPVYOVEVexjoA7Is5T8YX_vcnEpvQCCZyFCa0HInYIhoX46HCHR3HqGLsozv8GBpris_4lbW6c1yYzMOfAlBDA',
          'x-node-id': 'bt1bfnbm04r3qhdd2qdd',
          'x-folder-id': 'b1ghpjs6kvfc7th9878i',
          'Content-Type': 'application/json',
          Accept: '*/*',
          Connection: 'keep-alive',
          'Access-Control-Allow-Origin': '*',
          Origin: '*',
        },
      }
    );

    const result: ApiResponse = response.data;
    return result;
  } catch (error) {
    throw new Error(`HTTP error! status: ${error.response?.status || error.message}`);
  }
};
