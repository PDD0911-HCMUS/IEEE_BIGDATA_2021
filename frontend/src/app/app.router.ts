
import { Routes } from "@angular/router";
import { AppConsts } from "./services/app.consts";
import { ModelRunComponent } from "./run/model-run/model-run.component";

export class AppRouter {
  public static routes: Routes = [
    {
      path: AppConsts.page.model,
      component: ModelRunComponent,
    }
  ];
}
