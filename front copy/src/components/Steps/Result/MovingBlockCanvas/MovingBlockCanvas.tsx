/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable no-unsafe-optional-chaining */
//@ts-nocheck
import { useRef, useEffect, useState } from 'react';
import { Stage, Layer, Rect, Transformer, Image as KonvaImage } from 'react-konva';
import useImage from 'use-image';

import { Button, LoadingOverlay, Group } from '@mantine/core';
import { DownloadIcon, RefreshCcw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import lampSrc from '../../../../assets/Lamp.png';
import backgroundImageSrc from '../../../../assets/bg.jpeg';
import { useResult, useStepperForm } from '@/hooks';
import { bg, image } from './bg';

export const MovingBlockCanvas = () => {
  const [isDownlaoding, setIsDownloading] = useState(false);

  const { store } = useResult();
  const { stepperForm } = useStepperForm();
  const imageString =
    'data:image/png;base64,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';

  const stageRef = useRef();

  const rectRef = useRef();

  const imageRef = useRef();

  const trRef = useRef();

  const navigate = useNavigate();

  const [lampImage] = useImage(`data:image/png;base64,${image}`);
  const [backgroundImage] = useImage(`data:image/png;base64,${bg}`);

  const checkBoundaries = (node) => {
    const stage = node.getStage();
    const box = node.getClientRect();
    const width = stage.width();
    const height = stage.height();

    if (box.x < 0) {
      node.x(node.x() - box.x);
    }
    if (box.y < 0) {
      node.y(node.y() - box.y);
    }
    if (box.x + box.width > width) {
      node.x(node.x() - (box.x + box.width - width));
    }
    if (box.y + box.height > height) {
      node.y(node.y() - (box.y + box.height - height));
    }
  };

  useEffect(() => {
    const tr = trRef.current;
    if (tr) {
      tr.nodes([rectRef.current, imageRef.current].filter(Boolean));
      tr.getLayer().batchDraw();
    }
  }, [lampImage]);

  const getImagePositionAndScale = () => {
    if (!lampImage) return { x: 0, y: 0, scaleX: 1, scaleY: 1 };

    const rect = rectRef.current;
    const rectWidth = rect.width();
    const rectHeight = rect?.height();

    const imageAspectRatio = lampImage.width / lampImage.height;
    const rectAspectRatio = rectWidth / rectHeight;

    let scaleX;
    let scaleY;
    if (rectAspectRatio > imageAspectRatio) {
      scaleX = rectWidth / lampImage.width;
      scaleY = rectWidth / lampImage.width;
    } else {
      scaleX = rectHeight / lampImage.height;
      scaleY = rectHeight / lampImage.height;
    }

    const x = (rectWidth - lampImage.width * scaleX) / 2;
    const y = (rectHeight - lampImage.height * scaleY) / 2;

    return { x, y, scaleX, scaleY };
  };

  const downloadImage = () => {
    setIsDownloading(true);
    setTimeout(() => {
      const dataURL = stageRef.current.toDataURL();
      const link = document.createElement('a');
      link.download = `${stepperForm.productPicture?.fileName}_background.png`;
      link.href = dataURL;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setIsDownloading(false);
    }, 500);
  };

  return (
    <>
      <Group>
        <Button
          onClick={downloadImage}
          type="button"
          w="fit-content"
          leftSection={<DownloadIcon />}
          variant="outline"
        >
          Скачать изображение
        </Button>
        <Button type="button" w="fit-content" leftSection={<RefreshCcw />} variant="outline">
          Перегенерировать
        </Button>
      </Group>
      <LoadingOverlay
        visible={isDownlaoding}
        zIndex={1000}
        overlayProps={{ radius: 'sm', blur: 2 }}
        style={{ width: 514, height: 514, top: 114, left: '50%', transform: 'translate(-50%, 0)' }}
      />
      <Stage width={512} height={512} style={{ width: 512 }} ref={stageRef}>
        <Layer>
          {backgroundImage && <KonvaImage image={backgroundImage} width={512} height={512} />}
          <Rect
            x={9}
            y={75}
            width={lampImage?.width}
            height={lampImage?.height}
            draggable
            ref={rectRef}
            onDragMove={() => {
              checkBoundaries(rectRef.current);
            }}
            onTransform={() => {
              checkBoundaries(rectRef.current);
            }}
            onTransformEnd={() => {
              checkBoundaries(rectRef.current);
            }}
          />
          {lampImage && (
            <KonvaImage
              image={lampImage}
              x={rectRef.current?.x() + getImagePositionAndScale().x}
              y={rectRef.current?.y() + getImagePositionAndScale().y}
              scaleX={getImagePositionAndScale().scaleX}
              scaleY={getImagePositionAndScale().scaleY}
              draggable
              ref={imageRef}
              onDragMove={() => {
                checkBoundaries(imageRef.current);
              }}
              onTransform={() => {
                checkBoundaries(imageRef.current);
              }}
              onTransformEnd={() => {
                checkBoundaries(imageRef.current);
              }}
            />
          )}
          <Transformer
            ref={trRef}
            // borderStroke="black"
            borderStrokeWidth={1}
            boundBoxFunc={(oldBox, newBox) => {
              const stage = rectRef.current.getStage();
              const width = stage.width();
              const height = stage.height();

              if (newBox.x < 0 || newBox.y < 0) {
                return oldBox;
              }

              if (
                newBox.x < 0 ||
                newBox.y < 0 ||
                newBox.x + newBox.width > width ||
                newBox.y + newBox.height > height
              ) {
                return oldBox;
              }

              return newBox;
            }}
            enabledAnchors={['top-left', 'top-right', 'bottom-left', 'bottom-right']}
            borderEnabled={!isDownlaoding}
            resizeEnabled={!isDownlaoding}
            flipEnabled={!isDownlaoding}
            rotateEnabled={!isDownlaoding}
          />
        </Layer>
      </Stage>
    </>
  );
};
