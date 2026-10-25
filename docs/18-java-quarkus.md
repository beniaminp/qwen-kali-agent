# 18 — Java & Quarkus: versions, architecture and the model to train

*Last verified: 25 October 2026. Version facts were checked against the Quarkus releases page, the Quarkus 3.31/3.40/4.0 posts, OpenJDK and Oracle roadmaps.*

> If you want one Qwen3 base for Kali, Java and React instead of a Java-specific coder, see [26 — One Qwen3 for all](26-one-model-for-all.md). The Java-specific adapter here is still the higher-quality option for Java.

This is the start of the Java/Quarkus track. It answers three questions before any training:

1. **Which Java and Quarkus versions should the model target?**
2. **What does “Java architecture” mean in a Quarkus codebase, concretely?**
3. **Which Qwen coder can actually be fine-tuned on a 32 GB M1 Pro?**

The short answers:

- **Java 25 LTS + Quarkus 3.40 LTS** for production-shaped training data.
- **Hexagonal/DDD boundaries + SOLID + ArchUnit** as the architecture contract.
- **Qwen2.5-Coder-14B-Instruct 4-bit QLoRA** locally; **Qwen3-Coder-30B-A3B** as teacher/judge, not as a local training target.

---

## 1. Which “latest” do you target?

“Latest” is ambiguous in the Java ecosystem because there are two release trains: the six-month JDK train and the Quarkus LTS train.

| Component | Latest as of 25 Oct 2026 | Status | Use in this handbook |
|---|---|---|---|
| **Java 25** | 25.0.x | **LTS**, released 16 Sep 2025, supported to 2030 | **Primary target** |
| Java 27 | 27.0.x | Non-LTS, released 15 Sep 2026, supported to Mar 2027 | Feature previews / migration evals only |
| Java 21 | 21.0.x | Previous LTS | Legacy compatibility target |
| **Quarkus 3.40 LTS** | 3.40.1.1 (19 Oct 2026) | **LTS**, released 30 Sep 2026, supported to Sep 2027 | **Primary target** |
| Quarkus 3.33 LTS | 3.33.4.x | LTS, supported to Mar 2027 | Legacy migration source |
| Quarkus 4.0.0.Beta1 | Beta 1 (1 Oct 2026) | Beta; GA planned end of Nov 2026 | Migration evaluation only |
| Maven | 3.9.x | Required by Quarkus 3.31+ | Build tool |

### Why Java 25 LTS

Java 25 is the current long-term-support release. Quarkus 3.31 added **full Java 25 support**, including Java 25 runtime images and Mandrel native-image builds; 3.40 inherits that. Java 27 is a six-month, non-LTS release: it is useful for previewing features, but training a production assistant on Java 27-only idioms creates code that many teams cannot run.

### Why Quarkus 3.40 LTS

Quarkus 3.40 is the latest LTS and the recommended production line. It receives critical fixes and security patches for 12 months. Quarkus 4.0 is a major change (Java 21 baseline, Jakarta EE 11, Hibernate ORM 8, Jackson 3, CDI 5, Netty 4.2/Vert.x 5) and is still Beta as of this writing. Do not mix 4.0 idioms into a 3.40 adapter.

### The version policy for training data

```text
Primary:   Java 25 LTS + Quarkus 3.40 LTS + Maven 3.9 + Jakarta EE
Legacy:    Java 21 LTS + Quarkus 3.33 LTS
Migration: Quarkus 3.33 -> 3.40, Java 21 -> 25
Future:    Quarkus 4.0 migration tasks, evaluation only until GA
```

Tag every example with its target versions. A model trained on a mix of Quarkus 2 (`javax.*`), Quarkus 3 (`jakarta.*`) and Quarkus 4 (`Jackson 3`, Hibernate 8) without tags will produce a plausible but unbuildable blend.

---

## 2. Why Quarkus is a good fine-tuning target

Quarkus is opinionated in ways a model can learn:

- **Build-time augmentation.** Much of the wiring happens at build time, so the generated project shape matters.
- **CDI everywhere.** `@ApplicationScoped`, constructor injection, producers, interceptors, qualifiers.
- **RESTEasy Reactive / Quarkus REST.** `@Path`, `@GET`, `@POST`, `@Valid`, `Response`.
- **Panache.** Active Record or repository pattern over Hibernate ORM.
- **Configuration.** `application.properties` and `@ConfigMapping`.
- **Testing.** `@QuarkusTest`, RestAssured, Dev Services, `@QuarkusIntegrationTest`.
- **Native image.** `./mvnw package -Dnative`, Mandrel, reflection/resource registration.
- **Standards.** Jakarta REST, CDI, Bean Validation, JPA, JSON-B/Jackson, MicroProfile.

That gives you a finite, checkable idiom set: imports, annotations, package layout, test style, configuration keys. It is exactly the kind of thing a LoRA can learn from a few thousand high-quality examples.

Quarkus also has a large permissive corpus: the framework, quickstarts, guides and thousands of extensions are Apache-2.0. There is no ready-made “Quarkus instruction” dataset, but the raw material is unusually clean.

---

## 3. What the model must learn

| Layer | Skills | Examples |
|---|---|---|
| **Java language** | records, sealed types, pattern matching, generics, streams, `Optional`, exceptions, virtual threads, text blocks, try-with-resources | domain models, value objects, parsers, concurrency |
| **Quarkus** | CDI scopes, constructor injection, Quarkus REST, Panache, validation, config, security, messaging, scheduling, native | endpoints, repositories, DTOs, exception mappers, tests |
| **Architecture** | SOLID, hexagonal ports/adapters, DDD aggregates, package boundaries, coupling/cohesion | refactors, reviews, ArchUnit rules |
| **Engineering** | Maven, JUnit, RestAssured, Testcontainers, logging, error handling, transactions, migrations | patches, test suites, upgrade tasks |
| **Agent behaviour** | read before edit, patch format, run tests, diagnose failures, stop and ask | tool trajectories |

The model does **not** need to memorize the entire Quarkus reference. It needs to know the shape of a good Quarkus change and when to look things up.

---

## 4. Which Qwen coder?

### 4.1 The local recommendation

| Model | Params | Context | Local training on 32 GB | Role |
|---|---:|---:|---|---|
| **Qwen2.5-Coder-14B-Instruct** | 14.7B | 32K configured, 128K with RoPE scaling | **Yes**, 4-bit QLoRA, 24 of 48 blocks | **Primary local adapter** |
| Qwen2.5-Coder-7B-Instruct | 7.6B | 32K / 128K | Yes, all layers | Smoke test, fast iteration |
| Qwen2.5-Coder-32B-Instruct | 32.5B | 32K / 128K | Inference only at 4-bit | Occasional judge/teacher on a bigger machine |
| Qwen3-Coder-30B-A3B-Instruct | 30.5B total / 3.3B active | 256K native | **No** — inference/teacher only on 32 GB | Teacher, judge, bulk generation |
| Qwen3-Coder-Next | 80B total / 3B active | 256K | No | Too large for 32 GB; API/rented GPU |
| Qwen3-14B | 14B | 32K | Yes | General-agent fallback, less code-specialised |

### Why Qwen2.5-Coder-14B for local training

- It is **code-specialised** and strong on Java, not a general chat model that happens to write code.
- 14.7B at 4-bit is about **8.5–9 GB** of weights; a 24-block rank-16 adapter plus activations fits in 32 GB with gradient checkpointing.
- It has a **128K** context window (32K in the default config); you still train at 2K–4K because activation memory, not context length, is the constraint.
- It is **Apache-2.0** and has an MLX 4-bit conversion.
- It has no Gated DeltaNet training problem; it uses standard attention.

### Why not Qwen3-Coder-30B-A3B locally

Qwen3-Coder-30B-A3B is the better coder and agent, with 256K context and strong repository-level behaviour. But:

- 30.5B total parameters at 4-bit is roughly **18–20 GB** before activations.
- Training a 30B MoE with LoRA on 32 GB is not a supported, reliable MLX path; MoE routing and optimizer memory are the problems, not just weights.
- It is excellent as a **teacher/judge** served at 8K–16K context, or as a full fine-tune on a rented 80 GB GPU.

Use it to generate and grade Java/Quarkus data; train the 14B student locally.

### Why not Qwen3-Coder-Next

80B total, 3B active, 256K context. Even 4-bit weights are far beyond 32 GB. It is an API/rented-GPU model.

### Memory sketch

| Setup | Weights | Adapter + optimizer | Peak estimate |
|---|---:|---:|---:|
| Qwen2.5-Coder-7B 4-bit, all 28 blocks, rank 16 | ~4.5 GB | ~0.2 GB | 10–14 GB |
| Qwen2.5-Coder-14B 4-bit, 24/48 blocks, rank 16 | ~8.7 GB | ~0.4 GB | 18–23 GB |
| Qwen2.5-Coder-14B 4-bit, all 48 blocks, rank 16 | ~8.7 GB | ~0.8 GB | 21–26 GB (test) |
| Qwen3-Coder-30B-A3B 4-bit, inference only | ~18–20 GB | — | 22–28 GB at 8K–16K |

These are estimates. Measure with `mlx_lm.lora` and `mx.get_peak_memory()`; do not trust a table over your own machine.

---

## 5. What fine-tuning can and cannot do

**It can teach:**

- Quarkus-specific imports, annotations and package layout.
- The difference between `quarkus-rest` and legacy `quarkus-resteasy-reactive`.
- Constructor injection, records, validation, exception mapping.
- Panache repository vs Active Record, and where each belongs.
- SOLID/hexagonal refactors and ArchUnit rules.
- Your patch format, test command, and “read before edit” policy.
- Recovery from common Maven/Quarkus failures.

**It cannot:**

- Teach Java from zero; the base model already needs to code.
- Memorize a private repository; use retrieval or long context.
- Guarantee correctness; compile and test.
- Replace current documentation; APIs change.
- Learn Quarkus 4 before its idioms stabilize.

The right architecture is **model + repository context + compiler + tests + retrieval**, not a model that “knows” everything.

---

## 6. Java architecture for a Quarkus codebase

The most useful target is not “SOLID slogans” but a **package shape the model can reproduce and ArchUnit can check**.

### 6.1 Hexagonal / DDD layout

```text
src/main/java/com/example/orders/
├── domain/
│   ├── Order.java              # aggregate, invariants, no framework imports
│   ├── OrderLine.java          # value object
│   ├── CustomerId.java
│   ├── OrderStatus.java
│   ├── OrderRepository.java    # port (interface)
│   └── DomainException.java
├── application/
│   ├── PlaceOrderService.java  # use case, @ApplicationScoped, @Transactional
│   └── OrderQueryService.java
└── adapters/
    ├── rest/
    │   ├── OrderResource.java
    │   ├── PlaceOrderRequest.java
    │   ├── OrderResponse.java
    │   └── DomainExceptionMapper.java
    └── persistence/
        ├── OrderEntity.java
        ├── OrderLineEntity.java
        ├── OrderEntityRepository.java   # PanacheRepository
        └── PanacheOrderRepository.java  # implements domain port
```

Dependency rule:

```text
adapters → application → domain
```

The domain never imports `jakarta.persistence`, `io.quarkus`, `jakarta.ws.rs` or Panache. The application depends on domain ports. Adapters implement those ports and translate DTO/entity types.

### 6.2 SOLID, mapped to Quarkus

| Principle | In this codebase | Smell to reject |
|---|---|---|
| **S**RP | `OrderResource` does HTTP; `PlaceOrderService` does the use case; `PanacheOrderRepository` does persistence | a 400-line resource that validates, persists, maps and logs |
| **O**CP | new `DiscountPolicy` implementation instead of editing a `switch` | adding an `if` for every customer type |
| **L**SP | every `PaymentGateway` implementation returns the same result contract and throws the same domain errors | an implementation that returns `null` where others throw |
| **I**SP | small ports (`OrderReader`, `OrderWriter`, `PaymentGateway`) | one `OrderService` interface with 30 methods |
| **D**IP | domain declares `OrderRepository`; adapter implements it | domain imports `PanacheRepository` or `EntityManager` |

### 6.3 Quarkus idioms to train

```java
@ApplicationScoped
public class PlaceOrderService {
    private final OrderRepository orders;
    private final Clock clock;

    public PlaceOrderService(OrderRepository orders, Clock clock) { // constructor injection
        this.orders = orders;
        this.clock = clock;
    }

    @Transactional
    public Order place(CustomerId customerId, List<OrderLine> lines) {
        var order = Order.place(customerId, lines, clock.instant());
        return orders.save(order);
    }
}
```

- Constructor injection; no field injection.
- `@Transactional` on the use case, not the resource.
- Records for commands, DTOs and value objects.
- `@Valid` at the HTTP boundary; domain invariants in the aggregate.
- `ExceptionMapper` for domain errors.
- `@ConfigMapping` for typed configuration.
- `@QuarkusTest` + RestAssured for HTTP; plain JUnit for the domain.
- `quarkus-rest`, not `quarkus-resteasy-reactive`.
- `jakarta.*`, never `javax.persistence`/`javax.ws.rs`.

### 6.4 A minimal domain aggregate

```java
package com.example.orders.domain;

import java.time.Instant;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

public record Order(UUID id, CustomerId customerId, List<OrderLine> lines,
                    OrderStatus status, Instant createdAt) {

    public Order {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(customerId, "customerId");
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(createdAt, "createdAt");
        lines = List.copyOf(lines);
        if (lines.isEmpty()) {
            throw new DomainException("order must contain at least one line");
        }
    }

    public static Order place(CustomerId customerId, List<OrderLine> lines, Instant now) {
        return new Order(UUID.randomUUID(), customerId, lines, OrderStatus.PLACED, now);
    }
}
```

No `@Entity`, no `@ApplicationScoped`, no `jakarta.*`. It is a plain record you can test in milliseconds.

### 6.5 The port and the adapter

```java
package com.example.orders.domain;

import java.util.Optional;
import java.util.UUID;

public interface OrderRepository {
    Order save(Order order);
    Optional<Order> findById(UUID id);
}
```

```java
package com.example.orders.adapters.persistence;

import com.example.orders.domain.Order;
import com.example.orders.domain.OrderRepository;
import jakarta.enterprise.context.ApplicationScoped;

@ApplicationScoped
public class PanacheOrderRepository implements OrderRepository {
    private final OrderEntityRepository entities;

    public PanacheOrderRepository(OrderEntityRepository entities) {
        this.entities = entities;
    }

    @Override
    public Order save(Order order) {
        var entity = OrderMapper.toEntity(order);
        entities.persist(entity);
        return OrderMapper.toDomain(entity);
    }

    @Override
    public java.util.Optional<Order> findById(java.util.UUID id) {
        return entities.findByIdOptional(id).map(OrderMapper::toDomain);
    }
}
```

The JPA entity stays in `adapters.persistence`; the domain never sees it.

### 6.6 The HTTP boundary

```java
package com.example.orders.adapters.rest;

import jakarta.validation.Valid;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.net.URI;

@Path("/api/orders")
@Consumes(MediaType.APPLICATION_JSON)
@Produces(MediaType.APPLICATION_JSON)
public class OrderResource {

    private final PlaceOrderService service;

    public OrderResource(PlaceOrderService service) {
        this.service = service;
    }

    @POST
    public Response place(@Valid PlaceOrderRequest request) {
        var order = service.place(request.customerId(), request.lines());
        return Response.created(URI.create("/api/orders/" + order.id()))
                .entity(OrderResponse.from(order))
                .build();
    }
}
```

```java
public record PlaceOrderRequest(
        @jakarta.validation.constraints.NotBlank String customerId,
        @jakarta.validation.constraints.NotEmpty
        java.util.List<@Valid OrderLineRequest> lines) {
}
```

Use imports rather than fully-qualified annotations in real code; this is compressed for the page.

### 6.7 ArchUnit: make the architecture executable

```java
package com.example.orders;

import com.tngtech.archunit.core.importer.ImportOption;
import com.tngtech.archunit.junit.AnalyzeClasses;
import com.tngtech.archunit.junit.ArchTest;
import com.tngtech.archunit.lang.ArchRule;
import jakarta.enterprise.context.ApplicationScoped;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.*;
import static com.tngtech.archunit.library.dependencies.SlicesRuleDefinition.slices;

@AnalyzeClasses(packages = "com.example.orders", importOptions = ImportOption.DoNotIncludeTests.class)
class ArchitectureTest {

    @ArchTest
    static final ArchRule domain_is_independent = noClasses()
            .that().resideInAPackage("..domain..")
            .should().dependOnClassesThat()
            .resideInAnyPackage("..adapters..", "..application..", "jakarta..", "io.quarkus..");

    @ArchTest
    static final ArchRule services_are_application_scoped = classes()
            .that().resideInAPackage("..application..")
            .and().haveSimpleNameEndingWith("Service")
            .should().beAnnotatedWith(ApplicationScoped.class);

    @ArchTest
    static final ArchRule persistence_does_not_leak = noClasses()
            .that().resideOutsideOfPackage("..adapters.persistence..")
            .should().dependOnClassesThat().resideInAPackage("..adapters.persistence..");

    @ArchTest
    static final ArchRule no_package_cycles = slices()
            .matching("com.example.orders.(*)..")
            .should().beFreeOfCycles();
}
```

Add `archunit-junit5` in test scope. Run it with `./mvnw -q test`. For legacy code, `FreezingArchRule.freeze(rule)` lets you stop new violations without fixing everything at once.

### 6.8 Quarkus test

```java
@QuarkusTest
class OrderResourceTest {

    @Test
    void rejectsAnEmptyOrder() {
        given().contentType(ContentType.JSON)
                .body("""
                      {"customerId":"c-1","lines":[]}
                      """)
                .when().post("/api/orders")
                .then().statusCode(400);
    }
}
```

HTTP validation produces 400; domain errors mapped by `DomainExceptionMapper` can produce 422. Test both.

### 6.9 Maven baseline

Use the Quarkus BOM and do not pin extension versions individually:

```xml
<properties>
  <quarkus.platform.group-id>io.quarkus.platform</quarkus.platform.group-id>
  <quarkus.platform.version>3.40.1.1</quarkus.platform.version>
  <maven.compiler.release>25</maven.compiler.release>
  <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
</properties>

<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>${quarkus.platform.group-id}</groupId>
      <artifactId>quarkus-bom</artifactId>
      <version>${quarkus.platform.version}</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>

<dependencies>
  <dependency>
    <groupId>io.quarkus</groupId>
    <artifactId>quarkus-rest-jackson</artifactId>
  </dependency>
  <dependency>
    <groupId>io.quarkus</groupId>
    <artifactId>quarkus-hibernate-orm-panache</artifactId>
  </dependency>
  <dependency>
    <groupId>io.quarkus</groupId>
    <artifactId>quarkus-jdbc-postgresql</artifactId>
  </dependency>
  <dependency>
    <groupId>io.quarkus</groupId>
    <artifactId>quarkus-hibernate-validator</artifactId>
  </dependency>
  <dependency>
    <groupId>io.quarkus</groupId>
    <artifactId>quarkus-junit5</artifactId>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>io.rest-assured</groupId>
    <artifactId>rest-assured</artifactId>
    <scope>test</scope>
  </dependency>
  <dependency>
    <groupId>com.tngtech.archunit</groupId>
    <artifactId>archunit-junit5</artifactId>
    <version>1.5.1</version>
    <scope>test</scope>
  </dependency>
</dependencies>
```

The `quarkus-maven-plugin` executes `build`, `generate-code` and `generate-code-tests`; keep it at `${quarkus.platform.version}`. Use `./mvnw`, not a globally installed Maven, for reproducibility.

---

## 7. What the adapter should and should not learn

Train it to:

- produce the package shape above;
- choose `quarkus-rest` and `jakarta.*`;
- write the test before claiming success;
- return a diff that applies;
- explain trade-offs briefly.

Do **not** train it to:

- invent Quarkus configuration keys;
- use `javax.*` or Spring annotations;
- put `@Entity` on domain records;
- add dependencies silently;
- claim tests passed when it did not run them;
- output a 1,000-line “complete rewrite” for a 20-line change.

---

## 8. The data plan in one table

| Bucket | Share | Source |
|---|---:|---|
| Quarkus feature/API implementation | 25% | quickstarts, docs, synthetic |
| Repo bug fix / recovery | 20% | commits, SWE-smith-java, synthetic |
| SOLID/hexagonal refactor | 15% | synthetic + ArchUnit |
| Test generation | 15% | Methods2Test, synthetic |
| Code/architecture review | 10% | PR comments, synthetic |
| Migration/upgrade | 5% | 3.33→3.40, 21→25 |
| Refusal/uncertainty/security | 5% | synthetic negatives |
| General Java/code replay | 5% | OpenCodeInstruct, Magicoder |

Details, licenses and exact dataset IDs are in [19 — Java datasets](19-java-datasets.md). The training runbook is [20 — Java fine-tuning](20-java-finetuning.md). The synthetic pipeline is [21 — Java synthetic data](21-java-synthetic.md).

---

## 9. Sources

- Quarkus releases — <https://quarkus.io/releases>
- Quarkus 3.40 LTS — <https://quarkus.io/blog/quarkus-3-40-released/>
- Quarkus 3.31 (Java 25 support) — <https://quarkus.io/blog/quarkus-3-31-released/>
- Quarkus 4.0.0.Beta1 — <https://quarkus.io/blog/quarkus-4-0-0-beta1-released/>
- Quarkus 4 progress (Java 21 baseline) — <https://github.com/quarkusio/quarkus/discussions/55320>
- OpenJDK JDK 27 — <https://openjdk.org/projects/jdk/27/>
- Oracle Java SE support roadmap — <https://www.oracle.com/java/technologies/java-se-support-roadmap.html>
- Quarkus GitHub — <https://github.com/quarkusio/quarkus>
- Quarkus quickstarts — <https://github.com/quarkusio/quarkus-quickstarts>
- Qwen2.5-Coder-14B-Instruct — <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct>
- Qwen3-Coder-30B-A3B-Instruct — <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>
- Qwen3-Coder-Next — <https://huggingface.co/Qwen/Qwen3-Coder-Next>
- ArchUnit — <https://www.archunit.org/>
- Quarkus DDD/hexagonal — <https://quarkus.io/blog/quarkus-insights-254-ddd-hexagonal-architecture-part2>
