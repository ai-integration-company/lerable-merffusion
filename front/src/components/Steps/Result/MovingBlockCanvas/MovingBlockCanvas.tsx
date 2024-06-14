import React, { useEffect, useRef, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Group, Transformer } from 'react-konva';
import useImage from 'use-image';
import lamp from '../../../../assets/Lamp.png';

export const MovingBlockCanvas = () => {
  const stageRef = useRef();
  const groupRef = useRef();
  const transformerRef = useRef();
  const [stageCenterX, setStageCenterX] = useState(0);
  const [stageCenterY, setStageCenterY] = useState(0);
  const [selectedNode, setSelectedNode] = useState(null);
  const [images, setImages] = useState([]);
  const [bounceLocation, setBounceLocation] = useState({ x: 0, y: 0 });

  const initialImages = [{ x: 50, y: 50, width: 200, height: 200, src: lamp }];
  const [lampImage] = useImage(lamp);

  useEffect(() => {
    setImages(initialImages);
  }, []);

  useEffect(() => {
    setStageCenterX(600 / 2 - 400 / 2);
    setStageCenterY(600 / 2 - 300 / 2);
  }, []);

  const theyAreColliding = (r1, r2) =>
    !(
      r2.x > r1.x + r1.width ||
      r2.x + r2.width < r1.x ||
      r2.y > r1.y + r1.height ||
      r2.y + r2.height < r1.y
    );

  const handleDragEnd = (e, i) => {
    const { width, height } = e.target.size();
    const stage = e.target.getStage();
    const stageWidth = stage.width();
    const stageHeight = stage.height();

    let newX = e.target.x();
    let newY = e.target.y();

    if (newX < 0) newX = 0;
    if (newY < 0) newY = 0;
    if (newX > stageWidth - width) newX = stageWidth - width;
    if (newY > stageHeight - height) newY = stageHeight - height;

    const newImages = images.slice();
    newImages[i] = { ...newImages[i], x: newX, y: newY };
    setImages(newImages);
    setBounceLocation({ x: newX, y: newY });
  };

  return (
    <Stage
      ref={stageRef}
      height={600}
      width={600}
      style={{ border: '1px dashed gray' }}
      onClick={(e) => {
        if (e.target === stageRef.current) {
          transformerRef.current.nodes([]);
          setSelectedNode(null);
        }
      }}
    >
      <Layer>
        {/* <Group
          ref={groupRef}
          clipFunc={(ctx) => {
            ctx.rect(0, 0, 500, 400);
          }}
          x={stageCenterX}
          y={stageCenterY}
          height={400}
          width={500}
        >  */}
        {/* <Transformer
                ref={transformerRef}
                boundBoxFunc={(oldBox, newBox) => {
                  if (newBox.width < 50 || newBox.height < 50) {
                    return oldBox;
                  }
                  return newBox;
                }}
              > */}
        {/* <Rect x={0} y={0} height={400} width={500} fill="white" /> */}
        {images.map((img, i) => (
          <Group key={i} draggable x={img.x} y={img.y} width={img.width} height={img.height}>
            <Transformer
              ref={transformerRef}
              boundBoxFunc={(oldBox, newBox) => {
                if (newBox.width < 50 || newBox.height < 50) {
                  return oldBox;
                }
                return newBox;
              }}
            >
              <KonvaImage
                image={lampImage}
                x={0}
                y={0}
                width={img.width}
                height={img.height}
                onClick={() => {
                  setSelectedNode(i);
                  transformerRef.current.nodes([groupRef.current.children[i]]);
                }}
                onDragStart={() => {
                  transformerRef.current.nodes([groupRef.current.children[i]]);
                }}
                onDragEnd={(e) => handleDragEnd(e, i)}
              />
            </Transformer>
          </Group>
        ))}
        {/* </Transformer> */}
      </Layer>
    </Stage>
  );
};
