import { Hero } from "@/components/marketing/Hero";
import { Benchmarks } from "@/components/marketing/Benchmarks";
import { LiveLab } from "@/components/marketing/LiveLab";
import { ApiSection, Contact, CostCalc, FinalCta, Footer, Ownership, Pipeline, Workflows } from "@/components/marketing/Sections";

export default function Home() {
  return (
    <main>
      <Hero />
      <LiveLab />
      <Benchmarks />
      <Pipeline />
      <CostCalc />
      <Ownership />
      <Workflows />
      <ApiSection />
      <Contact />
      <FinalCta />
      <Footer />
    </main>
  );
}
