import { loadFont } from "@remotion/google-fonts/Inter";
import {
  AbsoluteFill,
  CalculateMetadataFunction,
  Composition,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const { fontFamily } = loadFont("normal", {
  weights: ["300", "700"],
  subsets: ["latin"],
});

type Props = {
  title: string;
  subtitle: string;
};

const calculateMetadata: CalculateMetadataFunction<Props> = () => {
  return {};
};

export const MyComposition = () => {
  return (
    <Composition
      id="Origins"
      component={MyComponent}
      durationInFrames={150}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={{ title: "Origins", subtitle: "A cosmic timeline" }}
      calculateMetadata={calculateMetadata}
    />
  );
};

export const MyComponent: React.FC<Props> = ({ title, subtitle }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const titleIn = spring({ frame, fps, config: { damping: 200 } });
  const subtitleIn = spring({
    frame,
    fps,
    delay: 15,
    config: { damping: 200 },
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 20, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  return (
    <AbsoluteFill
      style={{
        background:
          "radial-gradient(ellipse at center, #1b1446 0%, #05030f 70%)",
        justifyContent: "center",
        alignItems: "center",
        fontFamily,
        color: "white",
        opacity: fadeOut,
      }}
    >
      <div
        style={{
          fontSize: 180,
          fontWeight: 700,
          letterSpacing: interpolate(titleIn, [0, 1], [40, 4]),
          opacity: titleIn,
          transform: `scale(${interpolate(titleIn, [0, 1], [1.1, 1])})`,
        }}
      >
        {title}
      </div>
      <div
        style={{
          fontSize: 56,
          fontWeight: 300,
          opacity: subtitleIn * 0.8,
          transform: `translateY(${interpolate(subtitleIn, [0, 1], [30, 0])}px)`,
        }}
      >
        {subtitle}
      </div>
    </AbsoluteFill>
  );
};
