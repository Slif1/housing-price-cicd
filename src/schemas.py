from pydantic import BaseModel, Field, ValidationError


class HouseFeatures(BaseModel):
    MedInc: float = Field(ge=0)  # ge = greater or equal
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(ge=0)
    AveBedrms: float
    Population: float = Field(ge=0)
    AveOccup: float
    Latitude: float
    Longitude: float

    def to_array(self):
        return [
            self.MedInc,
            self.HouseAge,
            self.AveRooms,
            self.AveBedrms,
            self.Population,
            self.AveOccup,
            self.Latitude,
            self.Longitude,
        ]


if __name__ == "__main__":
    external_data = {
        "MedInc": "-18",
        "HouseAge": "7",
        "AveRooms": "-0.1",
        "Population": "-16",
    }
    try:
        HouseFeatures(**external_data)
    except ValidationError as e:
        print(e.errors())
