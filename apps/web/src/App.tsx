import { Shell } from "./components/Shell";
import { Analyze } from "./pages/Analyze";
import { Explore } from "./pages/Explore";
import { Benchmarks, Dataset, Method } from "./pages/Info";
import { ReportPage } from "./pages/ReportPage";
import { useRoute } from "./lib/router";

export function App() {
  const route = useRoute();
  return (
    <Shell current={route.page}>
      {route.page === "explore" && <Explore />}
      {route.page === "analyze" && <Analyze />}
      {route.page === "report" && <ReportPage key={`${route.source}-${route.id}`} source={route.source} id={route.id} />}
      {route.page === "dataset" && <Dataset />}
      {route.page === "benchmarks" && <Benchmarks />}
      {route.page === "method" && <Method />}
    </Shell>
  );
}
